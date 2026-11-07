# Training block: optimizer and train_angles contract

`main()` trains every circuit 10 times. Before each training it sets `SEED` to a
different value and calls `train_angles(qc, x_params, w_params, X, y)`, which must
return a 1-D numpy array of `len(w_params)` angles. Keep that name and signature
exactly. Draw the starting angles from `np.random.default_rng(SEED)`; a hard-coded
number makes all 10 trainings start from the same angles.
Inside, compute the loss with `cross_entropy(qc, x_params, w_params, w, X, y)`; the
protected training budget (178 passes over the training set) stops the search by
raising `BudgetSpent`, which `main()` catches - do not swallow it inside your own loop.
Valid scipy optimizers (`minimize` is already imported): `minimize(objective, w0,
method="COBYLA" | "Powell" | "Nelder-Mead" | "L-BFGS-B", options={"maxiter": MAX_ITER})`.
Angles are periodic, so wrap with `np.mod(w, 2 * np.pi)` rather than clipping.
