#include "utils_1d.h"
#include <mpi.h>
#include <limits.h>
#include <math.h>
#include <stddef.h>
#include <stdlib.h>

/* Algebraically equivalent explicit scheme for dt/dx^2 = 0.5. */
static inline double heat_step(const double *a, size_t j)
{
    return 0.5 * (a[j - 1] + a[j + 1]);
}

int heat_eqn_solver_1d(double *heat_array, double *err, size_t num_pts)
{
    int world_rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    if (heat_array == NULL || err == NULL || num_pts > INT_MAX)
        return -1;
    if (num_pts == 0)
        return 0;
const int active = num_pts < (size_t)world_size ? (int)num_pts : world_size;
    MPI_Comm comm = MPI_COMM_NULL;
    if (MPI_Comm_split(MPI_COMM_WORLD,
                       world_rank < active ? 1 : MPI_UNDEFINED,
                       world_rank, &comm) != MPI_SUCCESS)
        return -1;
    if (world_rank >= active)
        return 0;

    const int rank = world_rank;  /* Active ranks are a contiguous prefix. */
    const size_t base = num_pts / (size_t)active;
    const size_t rem = num_pts % (size_t)active;
    const size_t owned = base + ((size_t)rank < rem);
    const size_t first = 1u + (size_t)rank * base
                           + ((size_t)rank < rem ? (size_t)rank : rem);

    /* Up to 64 steps per halo exchange, but never more than one local block. */
    size_t h = base / 16u;
    if (h < 1u) h = 1u;
    if (h > 64u) h = 64u;
    if (h > base) h = base;
    if (active == 1) h = 1u;
    if (h > (size_t)ITER && ITER > 0) h = (size_t)ITER;
    if (h == 0u) h = 1u;

    const size_t owned_end = h + owned;  /* exclusive */
    const size_t len = owned + 2u * h;
    double *a = (double *)calloc(len, sizeof(double));
    double *b = (double *)calloc(len, sizeof(double));
    int *counts = NULL, *displs = NULL;

    if (rank == 0) {
        counts = (int *)malloc((size_t)active * sizeof(int));
        displs = (int *)malloc((size_t)active * sizeof(int));
    }

    const int failed = (a == NULL || b == NULL ||
                       (rank == 0 && (counts == NULL || displs == NULL)));
    int any_failed = 0;
    MPI_Allreduce(&failed, &any_failed, 1, MPI_INT, MPI_MAX, comm);
    if (any_failed) {
        free(a); free(b); free(counts); free(displs);
        MPI_Comm_free(&comm);
        return -1;
    }

    /* Every rank initially has the entire input array. */
    for (size_t k = 0; k < len; ++k) {
        const long long global = (long long)first + (long long)k - (long long)h;
        if (global >= 1 && global <= (long long)num_pts)
            a[k] = heat_array[(size_t)global];
    }

    /* Physical boundaries are never updated: both buffers keep them at zero. */
    const size_t valid_lo = (rank == 0) ? h : 0u;
    const size_t valid_end = (rank + 1 == active) ? owned_end : len;

    const double dx = 1.0 / (num_pts + 1);
    /* With dt = 0.5 * dx^2, the stencil coefficient is exactly 1/2. */
    double local_error = 0.0;

    for (size_t elapsed = 0; elapsed < (size_t)ITER; ) {
        const size_t remaining = (size_t)ITER - elapsed;
        const size_t steps = remaining < h ? remaining : h;

        if (elapsed > 0 && active > 1) {
            MPI_Request requests[4];
            int nreq = 0;
            if (rank > 0)
                MPI_Irecv(a, (int)h, MPI_DOUBLE, rank - 1, 100,
                          comm, &requests[nreq++]);
            if (rank + 1 < active)
                MPI_Irecv(a + owned_end, (int)h, MPI_DOUBLE, rank + 1, 101,
                          comm, &requests[nreq++]);
            if (rank > 0)
                MPI_Isend(a + h, (int)h, MPI_DOUBLE, rank - 1, 101,
                          comm, &requests[nreq++]);
            if (rank + 1 < active)
                MPI_Isend(a + owned_end - h, (int)h, MPI_DOUBLE, rank + 1, 100,
                          comm, &requests[nreq++]);
            MPI_Waitall(nreq, requests, MPI_STATUSES_IGNORE);
        }

        for (size_t s = 0; s < steps; ++s) {
            /* Shrinking space-time tile, expressed as half-open intervals. */
            size_t lo = h - steps + s + 1u;
            size_t end = owned_end + steps - s - 1u;
            if (lo < valid_lo) lo = valid_lo;
            if (end > valid_end) end = valid_end;

            /* Left halo: redundant computations, excluded from the error. */
            const size_t left_end = end < h ? end : h;
            for (size_t k = lo; k < left_end; ++k)
                b[k] = heat_step(a, k);

            /* Owned points: branch-free, SIMD reduction of cumulative error. */
            const size_t own_lo = lo > h ? lo : h;
            const size_t own_end = end < owned_end ? end : owned_end;
            double step_error = 0.0;
            #pragma omp simd reduction(+:step_error)
            for (size_t k = own_lo; k < own_end; ++k) {
                const double next = heat_step(a, k);
                b[k] = next;
                step_error += dx * fabs(next - a[k]);
            }
            local_error += step_error;

            /* Right halo: redundant computations, excluded from the error. */
            const size_t right_lo = lo > owned_end ? lo : owned_end;
            for (size_t k = right_lo; k < end; ++k)
                b[k] = heat_step(a, k);

            double *tmp = a; a = b; b = tmp;
        }
        elapsed += steps;
    }

    if (rank == 0) {
        for (int r = 0; r < active; ++r) {
            const size_t nr = base + ((size_t)r < rem);
            const size_t off = (size_t)r * base
                             + ((size_t)r < rem ? (size_t)r : rem);
            counts[r] = (int)nr;
            displs[r] = (int)off;
        }
    }

    double total_error = 0.0;
    MPI_Reduce(&local_error, &total_error, 1, MPI_DOUBLE, MPI_SUM, 0, comm);
    MPI_Gatherv(a + h, (int)owned, MPI_DOUBLE,
                rank == 0 ? heat_array + 1 : NULL,
                counts, displs, MPI_DOUBLE, 0, comm);
    if (rank == 0)
        *err += total_error;

    free(a); free(b); free(counts); free(displs);
    MPI_Comm_free(&comm);
    return 0;
}
