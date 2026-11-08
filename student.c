#include "utils_1d.h"
#include <mpi.h>
#include <limits.h>
#include <math.h>
#include <stddef.h>
#include <stdlib.h>

static inline double heat_step(const double *a, size_t j, double factor)
{
    return a[j] + factor * (a[j - 1] - 2.0 * a[j] + a[j + 1]);
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
//
    const int active = (num_pts < (size_t)world_size)
                         ? (int)num_pts : world_size;
    MPI_Comm comm = MPI_COMM_NULL;
    if (MPI_Comm_split(MPI_COMM_WORLD,
                       world_rank < active ? 1 : MPI_UNDEFINED,
                       world_rank, &comm) != MPI_SUCCESS)
        return -1;

    if (world_rank >= active)
        return 0;

    const int rank = world_rank;
    const size_t base = num_pts / (size_t)active;
    const size_t rem = num_pts % (size_t)active;
    const size_t owned = base + ((size_t)rank < rem ? 1u : 0u);
    const size_t first = 1u + (size_t)rank * base
                            + ((size_t)rank < rem ? (size_t)rank : rem);

    size_t h = base / 16u;
    if (h < 1u) h = 1u;
    if (h > 16u) h = 16u;
    if (h > base) h = base;
    if (active == 1) h = 1u;
    if (h > (size_t)ITER && ITER > 0) h = (size_t)ITER;
    if (h == 0u) h = 1u;

    const size_t len = owned + 2u * h;
    double *a = (double *)malloc(len * sizeof(double));
    double *b = (double *)malloc(len * sizeof(double));
    int *counts = NULL;
    int *displs = NULL;

    if (rank == 0) {
        counts = (int *)malloc((size_t)active * sizeof(int));
        displs = (int *)malloc((size_t)active * sizeof(int));
    }

    int failed = (a == NULL || b == NULL ||
                  (rank == 0 && (counts == NULL || displs == NULL)));
    int any_failed = 0;
    MPI_Allreduce(&failed, &any_failed, 1, MPI_INT, MPI_MAX, comm);
    if (any_failed) {
        free(a);
        free(b);
        free(counts);
        free(displs);
        MPI_Comm_free(&comm);
        return -1;
    }

    /* Every rank already has the initial array, including its halo. */
    for (size_t k = 0; k < len; ++k) {
        const long long global = (long long)first + (long long)k
                                 - (long long)h;
        a[k] = (global >= 1 && global <= (long long)num_pts)
                 ? heat_array[(size_t)global] : 0.0;
    }

    const double dx = 1.0 / (num_pts + 1);
    const double dt = 0.5 * dx * dx;
    const double factor = dt / (dx * dx);
    double local_error = 0.0;

    for (size_t elapsed = 0; elapsed < (size_t)ITER; ) {
        const size_t remaining = (size_t)ITER - elapsed;
        const size_t steps = remaining < h ? remaining : h;
        int overlap_computed = 0;

        if (rank == 0)
            for (size_t k = 0; k < h; ++k) a[k] = 0.0;
        if (rank + 1 == active)
            for (size_t k = h + owned; k < len; ++k) a[k] = 0.0;

        if (elapsed > 0 && active > 1) {
            /* Overlap halo exchange with independent interior updates. */
            MPI_Request requests[4];
            int nreq = 0;

            if (rank > 0)
                MPI_Irecv(a, (int)h, MPI_DOUBLE, rank - 1, 100,
                          comm, &requests[nreq++]);
            if (rank + 1 < active)
                MPI_Irecv(a + h + owned, (int)h, MPI_DOUBLE,
                          rank + 1, 101, comm, &requests[nreq++]);
            if (rank > 0)
                MPI_Isend(a + h, (int)h, MPI_DOUBLE, rank - 1, 101,
                          comm, &requests[nreq++]);
            if (rank + 1 < active)
                MPI_Isend(a + owned, (int)h, MPI_DOUBLE,
                          rank + 1, 100, comm, &requests[nreq++]);

            if (owned > 2u) {
                for (size_t k = h + 1u; k + 1u < h + owned; ++k) {
                    b[k] = heat_step(a, k, factor);
                    local_error += dx * fabs(b[k] - a[k]);
                }
                overlap_computed = 1;
            }
            MPI_Waitall(nreq, requests, MPI_STATUSES_IGNORE);
        }

        /* Shrinking spacetime tile; redundant halo updates do not affect err. */
        for (size_t s = 0; s < steps; ++s) {
            const size_t lo = h - steps + s + 1u;
            const size_t hi = h + owned + steps - s - 2u;
            for (size_t k = lo; k <= hi; ++k) {
                if (s == 0u && overlap_computed &&
                    k > h && k + 1u < h + owned)
                    continue;

                const long long global = (long long)first + (long long)k
                                         - (long long)h;
                if (global <= 0 || global > (long long)num_pts) {
                    b[k] = 0.0;
                } else {
                    b[k] = heat_step(a, k, factor);
                    if (k >= h && k < h + owned)
                        local_error += dx * fabs(b[k] - a[k]);
                }
            }
            double *tmp = a;
            a = b;
            b = tmp;
        }
        elapsed += steps;
    }

    if (rank == 0) {
        for (int r = 0; r < active; ++r) {
            const size_t nr = base + ((size_t)r < rem ? 1u : 0u);
            const size_t off = (size_t)r * base
                             + ((size_t)r < rem ? (size_t)r : rem);
            counts[r] = (int)nr;
            displs[r] = (int)off;
        }
    }

    double total_error = 0.0;
    MPI_Reduce(&local_error, &total_error, 1, MPI_DOUBLE,
               MPI_SUM, 0, comm);
    MPI_Gatherv(a + h, (int)owned, MPI_DOUBLE,
                rank == 0 ? heat_array + 1 : NULL,
                counts, displs, MPI_DOUBLE, 0, comm);
    if (rank == 0)
        *err += total_error;

    free(a);
    free(b);
    free(counts);
    free(displs);
    MPI_Comm_free(&comm);
    return 0;
}
