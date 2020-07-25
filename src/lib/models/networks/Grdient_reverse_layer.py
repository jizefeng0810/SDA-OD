from torch.autograd import Function


class ReverseLayerF(Function):

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha

        return output, None


# import torch
#
#
# class _GradientScalarLayer(torch.autograd.Function):
#     @staticmethod
#     def forward(ctx, input, weight):
#         ctx.weight = weight
#         return input.view_as(input)
#
#     @staticmethod
#     def backward(ctx, grad_output):
#         grad_input = grad_output.clone()
#         return ctx.weight*grad_input, None
#
# gradient_scalar = _GradientScalarLayer.apply
#
#
# class GradientScalarLayer(torch.nn.Module):
#     def __init__(self, weight):
#         super(GradientScalarLayer, self).__init__()
#         self.weight = weight
#
#     def forward(self, input):
#         return gradient_scalar(input, self.weight)
#
#     def __repr__(self):
#         tmpstr = self.__class__.__name__ + "("
#         tmpstr += "weight=" + str(self.weight)
#         tmpstr += ")"
#         return tmpstr