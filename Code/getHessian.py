def get_hessian(self, atoms, format="sparse", divide_by_masses=False):
    """
    Calculate the Hessian matrix for a pair potential.
    For an atomic configuration with N atoms in d dimensions the hessian matrix is a symmetric, hermitian matrix
    with a shape of (d*N,d*N). The matrix is in general a sparse matrix, which consists of dense blocks of
    shape (d,d), which are the mixed second derivatives. The result of the derivation for a pair potential can be
    found e.g. in:
    L. Pastewka et. al. "Seamless elastic boundaries for atomistic calculations", Phys. Rev. B 86, 075459 (2012).

    Parameters
    ----------
    atoms: ase.Atoms
        Atomic configuration in a local or global minima.

    format: "sparse" or "neighbour-list"
        Output format of the hessian matrix.

    divide_by_masses: bool
        if true return the dynamic matrix else hessian matrix

    Restrictions
    ----------
    This method is currently only implemented for three dimensional systems
    """
    if self.atoms is None:
        self.atoms = atoms

    f = self.f
    df = self.df
    df2 = self.df2

    nb_atoms = len(atoms)

    i_p, j_p, r_p, r_pc = neighbour_list("ijdD", atoms, self.dict)
    first_i = first_neighbours(nb_atoms, i_p)

    qi_p, qj_p = self._get_charges(i_p, j_p)

    e_p = np.zeros_like(r_p)
    de_p = np.zeros_like(r_p)
    dde_p = np.zeros_like(r_p)

    for mask, pair in self._mask_pairs(i_p, j_p):
        e_p[mask] = f[pair](r_p[mask], qi_p[mask], qj_p[mask])
        de_p[mask] = df[pair](r_p[mask], qi_p[mask], qj_p[mask])
        dde_p[mask] = df2[pair](r_p[mask], qi_p[mask], qj_p[mask])

    n_pc = r_pc / r_p[_c]
    nn_pcc = n_pc[..., :, np.newaxis] * n_pc[..., np.newaxis, :]
    H_pcc = -(dde_p[_cc] * nn_pcc)
    H_pcc += -((de_p / r_p)[_cc] * (np.eye(3, dtype=n_pc.dtype) - nn_pcc))

    # Sparse BSR-matrix
    if format == "sparse":
        if divide_by_masses:
            masses_n = atoms.get_masses()
            geom_mean_mass_p = np.sqrt(masses_n[i_p] * masses_n[j_p])
            H = bsr_matrix(
                ((H_pcc.T / geom_mean_mass_p).T, j_p, first_i),
                shape=(3 * nb_atoms, 3 * nb_atoms),
            )

        else:
            H = bsr_matrix(
                (H_pcc, j_p, first_i), shape=(3 * nb_atoms, 3 * nb_atoms)
            )

        Hdiag_icc = np.empty((nb_atoms, 3, 3))
        for x in range(3):
            for y in range(3):
                Hdiag_icc[:, x, y] = -np.bincount(
                    i_p, weights=H_pcc[:, x, y], minlength=nb_atoms
                )

        if divide_by_masses:
            H += bsr_matrix(
                (
                    (Hdiag_icc.T / masses_n).T,
                    np.arange(nb_atoms),
                    np.arange(nb_atoms + 1),
                ),
                shape=(3 * nb_atoms, 3 * nb_atoms),
            )

        else:
            H += bsr_matrix(
                (Hdiag_icc, np.arange(nb_atoms), np.arange(nb_atoms + 1)),
                shape=(3 * nb_atoms, 3 * nb_atoms),
            )

        return H

    # Neighbour list format
    elif format == "neighbour-list":
        return H_pcc, i_p, j_p, r_pc, r_p