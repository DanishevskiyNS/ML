def leave_one_out(model, X : np.array, y : np.array, h : float) -> int:
    correct_sum = 0

    for i in range(len(X)):
        x_test_ = X[i]
        y_test_ = y[i]
        X_ = np.delete(X, i, 0)
        y_ = np.delete(y, i, 0)
        model.fit(X_, y_, h)
        y_pred = model.predict_single_row(x_test_, h)
        if y_test_ == y_pred:
            correct_sum += 1

    return correct_sum
