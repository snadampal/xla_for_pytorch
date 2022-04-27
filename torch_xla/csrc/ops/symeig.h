#pragma once

#include "torch_xla/csrc/ir.h"

namespace torch_xla {

class SymEig : public XlaNode {
 public:
  SymEig(const XlaValue& input, bool eigenvectors, bool lower);

  std::string ToString() const override;

  torch::lazy::NodePtr Clone(OpList operands) const override;

  XlaOpVector Lower(LoweringContext* loctx) const override;

  bool eigenvectors() const { return eigenvectors_; }

  bool lower() const { return lower_; }

 private:
  bool eigenvectors_;
  bool lower_;
};

} // namespace torch_xla
