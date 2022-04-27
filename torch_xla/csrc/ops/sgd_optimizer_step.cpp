#include "torch_xla/csrc/ops/sgd_optimizer_step.h"

#include "torch_xla/csrc/lowering_context.h"
#include "torch_xla/csrc/ops/xla_ops.h"
#include "torch_xla/csrc/xla_lower_util.h"

namespace torch_xla {
namespace {

xla::Shape NodeOutputShape(const XlaValue& step, const XlaValue& param) {
  return xla::ShapeUtil::MakeTupleShape({/*step=*/step.xla_shape(),
                                         /*param=*/param.xla_shape(),
                                         /*buf=*/param.xla_shape()});
}

}  // namespace

SgdOptimizerStep::SgdOptimizerStep(
    const XlaValue& found_inf, const XlaValue& step, const XlaValue& param,
    const XlaValue& buf, const XlaValue& d_p, const XlaValue& weight_decay,
    const XlaValue& momentum, const XlaValue& lr, const XlaValue& dampening,
    bool use_weight_decay, bool use_momentum, bool use_nesterov)
    : XlaNode(xla_sgd_optimizer_step,
              {found_inf, step, param, buf, d_p, weight_decay, momentum, lr,
               dampening},
              NodeOutputShape(step, param),
              /*num_outputs=*/3,
              torch::lazy::MHash(use_weight_decay, use_momentum, use_nesterov)),
      use_weight_decay_(use_weight_decay),
      use_momentum_(use_momentum),
      use_nesterov_(use_nesterov) {}

torch::lazy::NodePtr SgdOptimizerStep::Clone(OpList operands) const {
  return torch::lazy::MakeNode<SgdOptimizerStep>(
      operands.at(0), operands.at(1), operands.at(2), operands.at(3),
      operands.at(4), operands.at(5), operands.at(6), operands.at(7),
      operands.at(8), use_weight_decay_, use_momentum_, use_nesterov_);
}

XlaOpVector SgdOptimizerStep::Lower(LoweringContext* loctx) const {
  xla::XlaOp found_inf = loctx->GetOutputOp(operand(0));
  xla::XlaOp step = loctx->GetOutputOp(operand(1));
  xla::XlaOp param = loctx->GetOutputOp(operand(2));
  xla::XlaOp buf = loctx->GetOutputOp(operand(3));
  xla::XlaOp d_p = loctx->GetOutputOp(operand(4));
  xla::XlaOp weight_decay = loctx->GetOutputOp(operand(5));
  xla::XlaOp momentum = loctx->GetOutputOp(operand(6));
  xla::XlaOp lr = loctx->GetOutputOp(operand(7));
  xla::XlaOp dampening = loctx->GetOutputOp(operand(8));
  return ReturnOps(
      BuildSgdOptimizerStep(found_inf, step, param, buf, d_p, weight_decay,
                            momentum, lr, dampening, use_weight_decay_,
                            use_momentum_, use_nesterov_),
      loctx);
}

} // namespace torch_xla
