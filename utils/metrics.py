import torch

def accuracy(y_true, y_pred, state=None):
    assert y_true.size() == y_pred.size() or \
           y_true.size() == y_pred.size()[:-1], \
           'y_true and y_pred have different size. ' \
           'y_true.size() = {}, y_pred.size() = {}'.format(
               y_true.size(), y_pred.size())

    if state is None:
        state = {'count': 0, 'total': 0}

    y_true = y_true.to(torch.device('cpu'))
    y_pred = y_pred.to(torch.device('cpu'))

    if y_true.size() == y_pred.size():
        correct = y_true == y_pred
    elif y_true.size() == y_pred.size()[:-1]:
        # Take the first occurance when there are multiple max values.
        # Should be compatible with numpy.argmax().
        n_class = y_pred.size()[-1]
        preds = n_class - 1 - y_pred.flip(-1).argmax(dim=-1)
        correct = y_true == preds

        # # Debug
        # preds_list = []
        # y_pred_list = y_pred.tolist()[0]
        # for p in y_pred_list:
        #     preds_list.append(max(enumerate(p), key=lambda x: x[1])[0])
        # assert preds.view(-1).tolist() == preds_list, \
        #        (preds, preds_list, y_pred, y_pred_list)

    state['count'] += correct.sum().item()
    state['total'] += correct.numel()
    value = state['count'] / state['total']

    return value, state
    
