import time


def train_with_gen(model, generator, criterion, optimizer,
                   max_iter=10):
    """
    This function wraps the training loop, just like keras.
    It assumes that generator yields x, y batches
    It does not support multiple criteria or optimizers
    """
    print('Starting training...')
    n_iter = 1
    record_loss = []
    tic = time.time()
    for x, y in generator:
        # cancel previous grad
        model.zero_grad()
        # forward
        y_pred = model.forward(x)
        # get the loss
        loss = criterion(y_pred, y)
        # store it
        record_loss.append(loss.data[0])
        # backward step
        loss.backward()
        # Optimization step
        optimizer.step()
        # increment
        print(n_iter, record_loss[-1], 'done in',
              time.time() - tic, 's')
        tic = time.time()
        n_iter += 1
        if n_iter > max_iter:
            break
    return model, record_loss
