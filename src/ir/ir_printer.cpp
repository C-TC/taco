#include <sstream>
#include <iostream>

#include "taco/ir/ir.h"
#include "taco/ir/ir_printer.h"
#include "taco/ir/simplify.h"
#include "taco/util/collections.h"
#include "taco/util/strings.h"

using namespace std;

namespace taco {
namespace ir {

const std::string magenta="\033[38;5;204m";
const std::string blue="\033[38;5;67m";
const std::string green="\033[38;5;70m";
const std::string orange="\033[38;5;214m";
const std::string nc="\033[0m";

template <class T>
static inline void acceptJoin(IRPrinter* printer, ostream& stream,
                              vector<T> nodes, string sep) {
  if (nodes.size() > 0) {
    nodes[0].accept(printer);
  }
  for (size_t i=1; i < nodes.size(); ++i) {
    stream << sep;
    nodes[i].accept(printer);
  }
}

IRPrinter::IRPrinter(ostream &s) : IRPrinter(s, false, false) {
}

IRPrinter::IRPrinter(ostream &s, bool color, bool simplify)
    : stream(s), indent(0), color(color), simplify(simplify) {
}

IRPrinter::~IRPrinter() {
}

void IRPrinter::setColor(bool color) {
  this->color = color;
}

void IRPrinter::print(Stmt stmt) {
  if (isa<Scope>(stmt)) {
    stmt = to<Scope>(stmt)->scopedStmt;
  }
  if (simplify) {
    Stmt oldStmt;
    do {
      oldStmt = stmt;
      stmt = ir::simplify(stmt);
    } while (stmt != oldStmt);
  }
  stmt.accept(this);
}

void IRPrinter::visit(const Literal* op) {
  if (color) {
    stream << blue ;
  }

  switch (op->type.getKind()) {
    case Datatype::Bool:
      stream << op->getValue<bool>();
    break;
    case Datatype::UInt8:
      stream << static_cast<uint16_t>(op->getValue<uint8_t>());
    break;
    case Datatype::UInt16:
      stream << op->getValue<uint16_t>();
    break;
    case Datatype::UInt32:
      stream << op->getValue<uint32_t>();
    break;
    case Datatype::UInt64:
      stream << op->getValue<uint64_t>();
    break;
    case Datatype::UInt128:
      taco_not_supported_yet;
    break;
    case Datatype::Int8:
      stream << static_cast<int16_t>(op->getValue<int8_t>());
    break;
    case Datatype::Int16:
      stream << op->getValue<int16_t>();
    break;
    case Datatype::Int32:
      stream << op->getValue<int32_t>();
    break;
    case Datatype::Int64:
      stream << op->getValue<int64_t>();
    break;
    case Datatype::Int128:
      taco_not_supported_yet;
    break;
    case Datatype::Float32:
      stream << ((op->getValue<float>() != 0.0)
                 ? util::toString(op->getValue<float>()) : "0.0");
    break;
    case Datatype::Float64:
      stream << ((op->getValue<double>()!=0.0)
                 ? util::toString(op->getValue<double>()) : "0.0");
    break;
    case Datatype::Complex64: {
      std::complex<float> val = op->getValue<std::complex<float>>();
      stream << val.real() << " + I*" << val.imag();
    }
    break;
    case Datatype::Complex128: {
      std::complex<double> val = op->getValue<std::complex<double>>();
      stream << val.real() << " + I*" << val.imag();
    }
    break;
    case Datatype::Undefined:
      taco_ierror << "Undefined type in IR";
    break;
  }

  if (color) {
    stream << nc;
  }
}

void IRPrinter::visit(const Var* op) {
  if (varNames.contains(op)) {
    stream << varNames.get(op);
  }
  else {
    stream << op->name;
  }
}

void IRPrinter::visit(const Neg* op) {
  if(op->type.isBool()) {
    stream << "!";
  } else {
    stream << "-";
  }
  parentPrecedence = Precedence::NEG;
  op->a.accept(this);
}

void IRPrinter::visit(const Sqrt* op) {
  stream << "sqrt(";
  op->a.accept(this);
  stream << ")";
}

void IRPrinter::visit(const Add* op) {
  printBinOp(op->a, op->b, "+", Precedence::ADD);
}

void IRPrinter::visit(const Sub* op) {
  printBinOp(op->a, op->b, "-", Precedence::SUB);
}

void IRPrinter::visit(const Mul* op) {
  printBinOp(op->a, op->b, "*", Precedence::MUL);
}

void IRPrinter::visit(const Div* op) {
  printBinOp(op->a, op->b, "/", Precedence::DIV);
}

void IRPrinter::visit(const Rem* op) {
  printBinOp(op->a, op->b, "%", Precedence::REM);
}

void IRPrinter::visit(const Min* op) {
  stream << "min(";
  for (size_t i=0; i<op->operands.size(); i++) {
    op->operands[i].accept(this);
    if (i < op->operands.size()-1)
      stream << ", ";
  }
  stream << ")";
}

void IRPrinter::visit(const Max* op) {
  stream << "max(";
  for (size_t i=0; i<op->operands.size(); i++) {
    op->operands[i].accept(this);
    if (i < op->operands.size()-1)
      stream << ", ";
  }
  stream << ")";
}


void IRPrinter::visit(const BitAnd* op){
  printBinOp(op->a, op->b, "&", Precedence::BAND);
}

void IRPrinter::visit(const BitOr* op){
  printBinOp(op->a, op->b, "|", Precedence::BOR);
}

void IRPrinter::visit(const Eq* op){
  printBinOp(op->a, op->b, "==", Precedence::EQ);
}

void IRPrinter::visit(const Neq* op) {
  printBinOp(op->a, op->b, "!=", Precedence::NEQ);
}

void IRPrinter::visit(const Gt* op) {
  printBinOp(op->a, op->b, ">", Precedence::GT);
}

void IRPrinter::visit(const Lt* op) {
  printBinOp(op->a, op->b, "<", Precedence::LT);
}

void IRPrinter::visit(const Gte* op) {
  printBinOp(op->a, op->b, ">=", Precedence::GTE);
}

void IRPrinter::visit(const Lte* op) {
  printBinOp(op->a, op->b, "<=", Precedence::LTE);
}

void IRPrinter::visit(const And* op) {
  printBinOp(op->a, op->b, keywordString("&&"), Precedence::LAND);
}

void IRPrinter::visit(const Or* op) {
  printBinOp(op->a, op->b, keywordString("||"), Precedence::LOR);
}

void IRPrinter::visit(const BinOp* op) {
  stream << op->strStart;
  op->a.accept(this);
  stream << op->strMid;
  op->b.accept(this);
  stream << op->strEnd;
}

void IRPrinter::visit(const Cast* op) {
  stream << "(" << keywordString(util::toString(op->type)) << ")";
  parentPrecedence = Precedence::CAST;
  op->a.accept(this);
}

void IRPrinter::visit(const Call* op) {
  stream << op->func << "(";
  parentPrecedence = Precedence::CALL;
  acceptJoin(this, stream, op->args, ", ");
  stream << ")";
}

void IRPrinter::visit(const IfThenElse* op) {
  taco_iassert(op->cond.defined());
  taco_iassert(op->then.defined());
  doIndent();
  stream << keywordString("if ");
  stream << "(";
  parentPrecedence = Precedence::TOP;
  op->cond.accept(this);
  stream << ")";

  Stmt scopedStmt = Stmt(to<Scope>(op->then)->scopedStmt);
  if (isa<Block>(scopedStmt)) {
    stream << " {" << endl;
    op->then.accept(this);
    doIndent();
    stream << "}";
  }
  else if (isa<Assign>(scopedStmt)) {
    int tmp = indent;
    indent = 0;
    stream << " ";
    scopedStmt.accept(this);
    indent = tmp;
  }
  else {
    stream << endl;
    op->then.accept(this);
  }

  if (op->otherwise.defined()) {
    stream << "\n";
    doIndent();
    stream << keywordString("else");
    stream << " {\n";
    op->otherwise.accept(this);
    doIndent();
    stream << "}";
  }
  stream << endl;
}

void IRPrinter::visit(const Case* op) {
  for (size_t i=0; i < op->clauses.size(); ++i) {
    auto clause = op->clauses[i];
    if (i != 0) stream << "\n";
    doIndent();
    if (i == 0) {
      stream << keywordString("if ");
      stream << "(";
      parentPrecedence = Precedence::TOP;
      clause.first.accept(this);
      stream << ")";
    }
    else if (i < op->clauses.size()-1 || !op->alwaysMatch) {
      stream << keywordString("else if ");
      stream << "(";
      parentPrecedence = Precedence::TOP;
      clause.first.accept(this);
      stream << ")";
    }
    else {
      stream << keywordString("else");
    }
    stream << " {\n";
    clause.second.accept(this);
    doIndent();
    stream << "}";
  }
  stream << endl;
}

void IRPrinter::visit(const Switch* op) {
  doIndent();
  stream << keywordString("switch ");
  stream << "(";
  op->controlExpr.accept(this);
  stream << ") {\n";
  indent++;
  for (const auto& switchCase : op->cases) {
    doIndent();
    stream << keywordString("case ");
    parentPrecedence = Precedence::TOP;
    switchCase.first.accept(this);
    stream << ": {\n";
    switchCase.second.accept(this);
    stream << "\n";
    indent++;
    doIndent();
    indent--;
    stream << keywordString("break");
    stream << ";\n";
    doIndent();
    stream << "}\n";
  }
  indent--;
  doIndent();
  stream << "}";
  stream << endl;
}

void IRPrinter::visit(const Load* op) {
  parentPrecedence = Precedence::LOAD;
  op->arr.accept(this);
  stream << "[";
  parentPrecedence = Precedence::LOAD;
  op->loc.accept(this);
  stream << "]";
}

void IRPrinter::visit(const Malloc* op) {
  stream << "malloc(";
  parentPrecedence = Precedence::TOP;
  op->size.accept(this);
  stream << ")";
}

void IRPrinter::visit(const Sizeof* op) {
  stream << "sizeof(";
  stream << op->sizeofType;
  stream << ")";
}

void IRPrinter::visit(const Store* op) {
  doIndent();
  op->arr.accept(this);
  stream << "[";
  parentPrecedence = Precedence::TOP;
  op->loc.accept(this);
  stream << "] = ";
  parentPrecedence = Precedence::TOP;
  op->data.accept(this);
  stream << ";";
  stream << endl;
}

void IRPrinter::visit(const For* op) {
  doIndent();
  stream << keywordString("for") << " (" 
         << keywordString(util::toString(op->var.type())) << " ";
  op->var.accept(this);
  stream << " = ";
  op->start.accept(this);
  stream << keywordString("; ");
  op->var.accept(this);
  stream << " < ";
  parentPrecedence = BOTTOM;
  op->end.accept(this);
  stream << keywordString("; ");
  op->var.accept(this);

  auto lit = op->increment.as<Literal>();
  if (lit != nullptr && ((lit->type.isInt()  && lit->equalsScalar(1)) ||
                         (lit->type.isUInt() && lit->equalsScalar(1)))) {
    stream << "++";
  }
  else {
    stream << " += ";
    op->increment.accept(this);
  }
  stream << ") {\n";

  op->contents.accept(this);
  doIndent();
  stream << "}";
  stream << endl;
}

void IRPrinter::visit(const While* op) {
  doIndent();
  stream << keywordString("while ");
  stream << "(";
  parentPrecedence = Precedence::TOP;
  op->cond.accept(this);
  stream << ")";
  stream << " {\n";
  op->contents.accept(this);
  doIndent();
  stream << "}";
  stream << endl;
}

void IRPrinter::visit(const Block* op) {
  acceptJoin(this, stream, op->contents, "");
}

void IRPrinter::visit(const Scope* op) {
  varNames.scope();
  indent++;
  op->scopedStmt.accept(this);
  indent--;
  varNames.unscope();
}

void IRPrinter::visit(const Function* op) {
  stream << keywordString("void ") << op->name;
  stream << "(";
  if (op->outputs.size() > 0) stream << "Tensor ";
  acceptJoin(this, stream, op->outputs, ", Tensor ");
  if (op->outputs.size() > 0 && op->inputs.size()) stream << ", ";
  if (op->inputs.size() > 0) stream << "Tensor ";
  acceptJoin(this, stream, op->inputs, ", Tensor ");
  stream << ") {" << endl;

  resetNameCounters();
  op->body.accept(this);

  doIndent();
  stream << "}";
}

void IRPrinter::visit(const VarDecl* op) {
  doIndent();
  stream << keywordString(util::toString(op->var.type()));
  taco_iassert(isa<Var>(op->var));
  if (to<Var>(op->var)->is_ptr) {
    stream << "* restrict";
  }
  stream << " ";
  string varName = varNameGenerator.getUniqueName(util::toString(op->var));
  varNames.insert({op->var, varName});
  op->var.accept(this);
  parentPrecedence = Precedence::TOP;
  stream << " = ";
  op->rhs.accept(this);
  stream << ";";
  stream << endl;
}

void IRPrinter::visit(const Assign* op) {
  doIndent();
  op->lhs.accept(this);
  parentPrecedence = Precedence::TOP;
  bool printed = false;
  if (simplify) {
    if (isa<ir::Add>(op->rhs)) {
      auto add = to<Add>(op->rhs);
      if (add->a == op->lhs) {
        const Literal* lit = add->b.as<Literal>();
        if (lit != nullptr && ((lit->type.isInt()  && lit->equalsScalar(1)) ||
                               (lit->type.isUInt() && lit->equalsScalar(1)))) {
          stream << "++";
        }
        else {
          stream << " += ";
          add->b.accept(this);
        }
        printed = true;
      }
    }
    else if (isa<Mul>(op->rhs)) {
      auto mul = to<Mul>(op->rhs);
      if (mul->a == op->lhs) {
        stream << " *= ";
        mul->b.accept(this);
        printed = true;
      }
    }
    else if (isa<BitOr>(op->rhs)) {
      auto bitOr = to<BitOr>(op->rhs);
      if (bitOr->a == op->lhs) {
        stream << " |= ";
        bitOr->b.accept(this);
        printed = true;
      }
    }
  }
  if (!printed) {
    stream << " = ";
    op->rhs.accept(this);
  }

  stream << ";";
  stream << endl;
}

void IRPrinter::visit(const Yield* op) {
  doIndent();
  stream << "yield({";
  acceptJoin(this, stream, op->coords, ", ");
  stream << "}, ";
  op->val.accept(this);
  parentPrecedence = Precedence::TOP;
  stream << ");";
  stream << endl;
}

void IRPrinter::visit(const Allocate* op) {
  doIndent();
  if (op->is_realloc)
    stream << "reallocate ";
  else
    stream << "allocate ";
  op->var.accept(this);
  stream << "[";
  op->num_elements.accept(this);
  stream << "]";
  stream << endl;
}

void IRPrinter::visit(const Free* op) {
  doIndent();
  stream << "free(";
  parentPrecedence = Precedence::TOP;
  op->var.accept(this);
  stream << ");";
  stream << endl;
}

void IRPrinter::visit(const Comment* op) {
  doIndent();
  stream << commentString(op->text);
  stream << endl;
}

void IRPrinter::visit(const BlankLine*) {
  stream << endl;
}

void IRPrinter::visit(const Continue*) {
  doIndent();
  stream << "continue;" << endl;
}

void IRPrinter::visit(const Break*) {
  doIndent();
  stream << "break;" << endl;
}

void IRPrinter::visit(const Print* op) {
  doIndent();
  stream << "printf(";
  stream << "\"" << op->fmt << "\"";
  for (auto e: op->params) {
    stream << ", ";
    e.accept(this);
  }
  stream << ");";
  stream << endl;
}

void IRPrinter::visit(const GetProperty* op) {
  stream << op->name;
}

void IRPrinter::visit(const Sort* op) {
  doIndent();
  stream << "qsort(";
  parentPrecedence = Precedence::CALL;
  acceptJoin(this, stream, op->args, ", ");
  stream << ", cmp);";
  stream << endl;
}


void IRPrinter::resetNameCounters() {
  // seed the unique names with all C99 keywords
  // from: http://en.cppreference.com/w/c/keyword
  vector<string> keywords =
    {"auto",
     "break",
     "case",
     "char",
     "const",
     "continue",
     "default",
     "do",
     "double",
     "else",
     "enum",
     "extern",
     "float",
     "for",
     "goto",
     "if",
     "inline",
     "int",
     "long",
     "register",
     "restrict",
     "return",
     "short",
     "signed",
     "sizeof",
     "static",
     "struct",
     "switch",
     "typedef",
     "union",
     "unsigned",
     "void",
     "volatile",
     "while",
     "bool",
     "complex",
     "imaginary"};
  varNameGenerator = util::NameGenerator(keywords);
}

void IRPrinter::doIndent() {
  for (int i=0; i<indent; i++)
    stream << "  ";
}

void IRPrinter::printBinOp(Expr a, Expr b, string op, Precedence precedence) {
  bool parenthesize = needsParentheses(precedence);
  if (parenthesize) {
    stream << "(";
  }
  parentPrecedence = precedence;
  a.accept(this);
  stream << " " << op << " ";
  parentPrecedence = precedence;
  b.accept(this);
  if (parenthesize) {
    stream << ")";
  }
}

bool IRPrinter::needsParentheses(Precedence precedence) {
  // Add parentheses if required by C operator precedence or for Boolean
  // expressions of form `a || (b && c)` (to avoid C compiler warnings)
  return (precedence >= parentPrecedence ||
            (precedence == Precedence::LAND &&
              parentPrecedence == Precedence::LOR));
}


std::string IRPrinter::keywordString(std::string keyword) {
  if (color) {
    return magenta + keyword + nc;
  }
  else {
    return keyword;
  }
}

std::string IRPrinter::commentString(std::string comment) {
  if (color) {
    return green + "/* " + comment + " */" + nc;
  }
  else {
    return "/* " + comment + " */";
  }
}

DacePrinter::DacePrinter(std::ostream& stream)
    : IRPrinter(stream, false, true) {
}

DacePrinter::DacePrinter(std::ostream& stream, bool color, bool simplify, vector<std::string> sampled_replace)
    : IRPrinter(stream, color, simplify), sampled_replace(sampled_replace) {
}

DacePrinter::~DacePrinter() {
}

void DacePrinter::print(Stmt stmt) {
  if (isa<Scope>(stmt)) {
    stmt = to<Scope>(stmt)->scopedStmt;
  }
  if (simplify) {
    Stmt oldStmt;
    do {
      oldStmt = stmt;
      stmt = ir::simplify(stmt);
    } while (stmt != oldStmt);
  }

  // print python import
  stream << "import dace" << endl;
  stream << "import numpy as np" << endl;
  stream << "from math import sqrt" << endl;

  struct PropertyCollector: public IRVisitor {
    map<string, Datatype::Kind> arrays;
    set<string> symbols;
    Datatype vals_type;
    using IRVisitor::visit;
    void visit(const GetProperty* op) {
      switch (op->property)
      {
      case TensorProperty::Dimension:
        symbols.insert(op->name);
        break;
      case TensorProperty::Indices:
        arrays[op->name] = op->tensor.type().getKind();
        break;
      case TensorProperty::Values:
        arrays[op->name] = op->tensor.type().getKind();
        vals_type = max_type(vals_type, arrays[op->name]);
        break;
      default:
        taco_not_supported_yet;
      }
    }
  };
  PropertyCollector coll;
  stmt.accept(&coll);
  vals_type = coll.vals_type;
  arrays = coll.arrays;

  
  for (const string &sym : coll.symbols){
    stream << sym << " = dace.symbol('" << sym << "')" << endl;
  }

  for (const auto &arr : coll.arrays){
    stream << "size_" << arr.first << " = dace.symbol('" << "size_" << arr.first << "')" << endl;
  }

  // sampled replace
  // collect jA, jD from A, D
  // assume jA and jD are Var.
  struct ValsLocCollector: public IRVisitor {
    Var* src_var = nullptr;
    Var*  target_var = nullptr;
    vector<std::string> sampled_replace;
    ValsLocCollector(vector<std::string> sampled_replace) : sampled_replace(sampled_replace) {}

    using IRVisitor::visit;

    void visit(const Load* op) {
      if (isa<GetProperty>(op->arr)) {
        auto get_prop = to<GetProperty>(op->arr);
        if (get_prop->property == TensorProperty::Values) {
          if (isa<Var>(get_prop->tensor)) {
            auto tensor_var = to<Var>(get_prop->tensor);
            if (tensor_var->name == sampled_replace[0]) {
              src_var = const_cast<Var*>(to<Var>(op->loc));
            } else if (tensor_var->name == sampled_replace[1]) {
              target_var = const_cast<Var*>(to<Var>(op->loc));
            }
          }
        }
      }
    }

    void visit(const Store* op) {
      if (isa<GetProperty>(op->arr)) {
        auto get_prop = to<GetProperty>(op->arr);
        if (get_prop->property == TensorProperty::Values) {
          if (isa<Var>(get_prop->tensor)) {
            auto tensor_var = to<Var>(get_prop->tensor);
            if (tensor_var->name == sampled_replace[0]) {
              src_var = const_cast<Var*>(to<Var>(op->loc));
            } else if (tensor_var->name == sampled_replace[1]) {
              target_var = const_cast<Var*>(to<Var>(op->loc));
            }
          }
        }
      }
      op->data.accept(this);
    }


  };

  if (!sampled_replace.empty()) {
    ValsLocCollector var_coll(sampled_replace);
    stmt.accept(&var_coll);
    src_var = var_coll.src_var;
    target_var = var_coll.target_var;
    if (src_var == nullptr || target_var == nullptr) {
      taco_ierror << "Can not find sampled replace index pair." << endl;
    }
  }

  stmt.accept(this);
}

void DacePrinter::visit(const Literal* op) {
  if (color) {
    stream << blue;
  }

  switch (op->type.getKind()) {
    case Datatype::Bool:
      stream << (op->getValue<bool>()? "True" : "False");
      break;
    case Datatype::UInt8:
      stream << static_cast<int64_t>(op->getValue<uint8_t>()); 
      break;
    case Datatype::UInt16:
      stream << static_cast<int64_t>(op->getValue<uint16_t>()); 
      break;
    case Datatype::UInt32:
      stream << static_cast<int64_t>(op->getValue<uint32_t>()); 
      break;
    case Datatype::UInt64:
      taco_not_supported_yet;
      break;
    case Datatype::UInt128:
      taco_not_supported_yet;
      break;
    case Datatype::Int8:
      stream << static_cast<int64_t>(op->getValue<int8_t>());
      break;
    case Datatype::Int16:
      stream << static_cast<int64_t>(op->getValue<int16_t>());
      break;
    case Datatype::Int32:
      stream << static_cast<int64_t>(op->getValue<int32_t>());
      break;
    case Datatype::Int64:
      stream << op->getValue<int64_t>();
      break;
    case Datatype::Int128:
      taco_not_supported_yet;
      break;
    case Datatype::Float32:
      stream << ((op->getValue<float>() != 0.0)
                 ? util::toString(op->getValue<float>()) : "0.0");
      break;
    case Datatype::Float64:
      stream << ((op->getValue<double>() != 0.0)
                 ? util::toString(op->getValue<double>()) : "0.0");
      break;
    case Datatype::Complex64:
      stream << op->getValue<std::complex<float>>().real() << " + "
             << op->getValue<std::complex<float>>().imag() << "j";
      break;
    case Datatype::Complex128:
      stream << op->getValue<std::complex<double>>().real() << " + "
             << op->getValue<std::complex<double>>().imag() << "j";
      break;
    case Datatype::Undefined:
      taco_ierror << "Undefined datatype";
      break;
  }

  if (color) {
    stream << nc;
  }
}

void DacePrinter::visit(const Var* op) {
  // SDDMM jA->jD
  if (src_var == op) {
    this->visit(target_var);
    return;
  }

  if (varNames.contains(op)) {
    stream << varNames.get(op);
  }
  else {
    stream << op->name;
  }
}

void DacePrinter::visit(const Neg* op) {
  if(op->type.isBool()) {
    stream << "not ";
  } else {
    stream << "-";
  }
  parentPrecedence = Precedence::NEG;
  op->a.accept(this);
}

void DacePrinter::visit(const Sqrt* op) {
  stream << "sqrt(";
  op->a.accept(this);
  stream << ")";
}

void DacePrinter::visit(const Add* op) {
  printBinOp(op->a, op->b, "+", Precedence::ADD);
}

void DacePrinter::visit(const Sub* op) {
  printBinOp(op->a, op->b, "-", Precedence::SUB);
}

void DacePrinter::visit(const Mul* op) {
  printBinOp(op->a, op->b, "*", Precedence::MUL);
}

void DacePrinter::visit(const Div* op) {
  printBinOp(op->a, op->b, "/", Precedence::DIV);
}

void DacePrinter::visit(const Rem* op) {
  printBinOp(op->a, op->b, "%", Precedence::REM);
}

void DacePrinter::visit(const Min* op) {
  stream << "min(";
  for (size_t i=0; i<op->operands.size(); i++) {
    op->operands[i].accept(this);
    if (i < op->operands.size()-1)
      stream << ", ";
  }
  stream << ")";
}

void DacePrinter::visit(const Max* op) {
  stream << "max(";
  for (size_t i=0; i<op->operands.size(); i++) {
    op->operands[i].accept(this);
    if (i < op->operands.size()-1)
      stream << ", ";
  }
  stream << ")";
}

void DacePrinter::visit(const BitAnd* op){
  printBinOp(op->a, op->b, "&", Precedence::BAND);
}

void DacePrinter::visit(const BitOr* op){
  printBinOp(op->a, op->b, "|", Precedence::BOR);
}

void DacePrinter::visit(const Eq* op){
  printBinOp(op->a, op->b, "==", Precedence::EQ);
}

void DacePrinter::visit(const Neq* op) {
  printBinOp(op->a, op->b, "!=", Precedence::NEQ);
}

void DacePrinter::visit(const Gt* op) {
  printBinOp(op->a, op->b, ">", Precedence::GT);
}

void DacePrinter::visit(const Lt* op) {
  printBinOp(op->a, op->b, "<", Precedence::LT);
}

void DacePrinter::visit(const Gte* op) {
  printBinOp(op->a, op->b, ">=", Precedence::GTE);
}

void DacePrinter::visit(const Lte* op) {
  printBinOp(op->a, op->b, "<=", Precedence::LTE);
}

void DacePrinter::visit(const And* op) {
  printBinOp(op->a, op->b, keywordString(" and "), Precedence::LAND);
}

void DacePrinter::visit(const Or* op) {
  printBinOp(op->a, op->b, keywordString(" or "), Precedence::LOR);
}

void DacePrinter::visit(const BinOp* op) {
  stream << op->strStart;
  op->a.accept(this);
  stream << op->strMid;
  op->b.accept(this);
  stream << op->strEnd;
}

void DacePrinter::visit(const Cast* op) {
  // casting in python: int, float, complex
  if (op->type.isInt()) {
    stream << "int(";
  } else if (op->type.isFloat()) {
    stream << "float(";
  } else if (op->type.isComplex()) {
    stream << "complex(";
  } else {
    taco_ierror << "Unsupported cast type";
  }
  parentPrecedence = Precedence::CAST;
  op->a.accept(this);
  stream << ")";
}

void DacePrinter::visit(const Call* op) {
  stream << op->func << "(";
  parentPrecedence = Precedence::CALL;
  acceptJoin(this, stream, op->args, ", ");
  stream << ")";
}


void DacePrinter::visit(const IfThenElse* op) {
  taco_iassert(op->cond.defined());
  taco_iassert(op->then.defined());
  doIndent();
  stream << keywordString("if ");
  parentPrecedence = Precedence::TOP;
  op->cond.accept(this);

  stream << " :" << endl;
  op->then.accept(this);
  doIndent();

  if (op->otherwise.defined()) {
    stream << "\n";
    doIndent();
    stream << keywordString("else:");
    stream << "\n";
    op->otherwise.accept(this);
    doIndent();
  }
  stream << endl;
}


void DacePrinter::visit(const Case* op) {
  for (size_t i=0; i < op->clauses.size(); ++i) {
    auto clause = op->clauses[i];
    if (i != 0) stream << "\n";
    doIndent();
    if (i == 0) {
      stream << keywordString("if ");
      parentPrecedence = Precedence::TOP;
      clause.first.accept(this);
    }
    else if (i < op->clauses.size()-1 || !op->alwaysMatch) {
      stream << keywordString("elif ");
      parentPrecedence = Precedence::TOP;
      clause.first.accept(this);
    }
    else {
      stream << keywordString("else");
    }
    stream << " :\n";
    clause.second.accept(this);
    doIndent();
  }
  stream << endl;
}

void DacePrinter::visit(const Switch* op) {
  taco_not_supported_yet;
}


void DacePrinter::visit(const Load* op) {
  parentPrecedence = Precedence::LOAD;
  op->arr.accept(this);
  stream << "[";
  parentPrecedence = Precedence::LOAD;
  op->loc.accept(this);
  stream << "]";
}

string match_numpy_type(const Datatype::Kind kind) {
  switch (kind) {
    case Datatype::Bool: return "bool_";
    case Datatype::UInt8: return "uint8";
    case Datatype::UInt16: return "uint16"; 
    case Datatype::UInt32: return "uint32";
    case Datatype::UInt64: return "uint64";
    case Datatype::UInt128: taco_not_supported_yet; break;
    case Datatype::Int8: return "int8";
    case Datatype::Int16: return "int16";
    case Datatype::Int32: return "int32";
    case Datatype::Int64: return "int64";
    case Datatype::Int128: taco_not_supported_yet; break;
    case Datatype::Float32: return "float32";
    case Datatype::Float64: return "float64";
    case Datatype::Complex64: return "complex64";
    case Datatype::Complex128: return "complex128";
    case Datatype::Undefined: taco_not_supported_yet; break;
  }
}

void DacePrinter::visit(const Malloc* op) {
  stream << "np.zeros(";
  parentPrecedence = Precedence::TOP;
  op->size.accept(this);
  stream << ", dtype=np." << match_numpy_type(vals_type.getKind()) << ")";
}

void DacePrinter::visit(const Sizeof* op) {
  // hack: assume Sizeof is only used for Malloc
  stream << "1";
}

void DacePrinter::visit(const Store* op) {
  doIndent();
  op->arr.accept(this);
  stream << "[";
  parentPrecedence = Precedence::TOP;
  op->loc.accept(this);
  stream << "] = ";
  parentPrecedence = Precedence::TOP;
  op->data.accept(this);
  stream << endl;
}


void DacePrinter::visit(const For* op) {
  doIndent();
  stream << keywordString("for") << " ";
  op->var.accept(this);
  stream << " in range(";
  op->start.accept(this);
  stream << keywordString(", ");
  parentPrecedence = BOTTOM;
  op->end.accept(this);
  stream << keywordString(", ");
  op->increment.accept(this);
  stream << "):\n";

  op->contents.accept(this);
  doIndent();
  stream << endl;
}

void DacePrinter::visit(const While* op) {
  doIndent();
  stream << keywordString("while ");
  parentPrecedence = Precedence::TOP;
  op->cond.accept(this);
  stream << ":\n";
  op->contents.accept(this);
  doIndent();
  stream << endl;
}

void DacePrinter::visit(const Block* op) {
  acceptJoin(this, stream, op->contents, "");
}

void DacePrinter::visit(const Scope* op) {
  varNames.scope();
  indent++;
  op->scopedStmt.accept(this);
  indent--;
  varNames.unscope();
}

void DacePrinter::visit(const Function* op) {
  stream << "@dace.program\n";
  stream << keywordString("def ") << op->name;
  stream << "(";
  
  for (auto it = arrays.begin(); it != arrays.end(); it++) {
    if (it != arrays.begin()) stream << ", ";
    stream << it->first << ": dace." << match_numpy_type(it->second) << "[size_" << it->first << "]";
  }
  stream << "):" << endl;

  resetNameCounters();
  op->body.accept(this);

  doIndent();
}

void DacePrinter::visit(const VarDecl* op) {
  taco_iassert(isa<Var>(op->var));
  // SDDMM
  if (to<Var>(op->var) == src_var) {
    return;
  }


  doIndent();
  
  if (to<Var>(op->var)->is_ptr) {
    ;
  }
  
  string varName = varNameGenerator.getUniqueName(util::toString(op->var));
  varNames.insert({op->var, varName});
  op->var.accept(this);
  parentPrecedence = Precedence::TOP;
  stream << " = ";
  op->rhs.accept(this);
  stream << endl;
}


void DacePrinter::visit(const Assign* op) {
  // SDDMM
  if (isa<Var>(op->lhs) && to<Var>(op->lhs) == src_var) {
    return;
  }

  doIndent();
  op->lhs.accept(this);
  parentPrecedence = Precedence::TOP;
  bool printed = false;
  if (simplify) {
    if (isa<ir::Add>(op->rhs)) {
      auto add = to<Add>(op->rhs);
      if (add->a == op->lhs) {
        stream << " += ";
        add->b.accept(this);
        printed = true;
      }
    }
    else if (isa<Mul>(op->rhs)) {
      auto mul = to<Mul>(op->rhs);
      if (mul->a == op->lhs) {
        stream << " *= ";
        mul->b.accept(this);
        printed = true;
      }
    }
    else if (isa<BitOr>(op->rhs)) {
      auto bitOr = to<BitOr>(op->rhs);
      if (bitOr->a == op->lhs) {
        stream << " |= ";
        bitOr->b.accept(this);
        printed = true;
      }
    }
  }
  if (!printed) {
    stream << " = ";
    op->rhs.accept(this);
  }

  stream << endl;
}

void DacePrinter::visit(const Yield* op) {
  taco_not_supported_yet;
}

void DacePrinter::visit(const Allocate* op) {
  doIndent();

  op->var.accept(this);
  stream << " = np.zeros(";
  parentPrecedence = Precedence::TOP;
  op->num_elements.accept(this);
  stream << ")";
  stream << endl;
}

void DacePrinter::visit(const Free* op) {
  doIndent();
  stream << endl;
}


void DacePrinter::visit(const Comment* op) {
  doIndent();
  stream << commentString(op->text);
  stream << endl;
}

void DacePrinter::visit(const BlankLine*) {
  stream << endl;
}

void DacePrinter::visit(const Continue*) {
  doIndent();
  stream << "continue" << endl;
}

void DacePrinter::visit(const Break*) {
  doIndent();
  stream << "break" << endl;
}


void DacePrinter::visit(const Print* op) {
  doIndent();
  stream << "# print(";
  stream << "\"" << op->fmt << "\"";
  for (auto e: op->params) {
    stream << ", ";
    e.accept(this);
  }
  stream << ");";
  stream << endl;
}

void DacePrinter::visit(const GetProperty* op) {
  stream << op->name;
}

void DacePrinter::visit(const Sort* op) {
  doIndent();
  taco_not_supported_yet;
  stream << endl;
}


void DacePrinter::resetNameCounters() {
  vector<string> keywords =
    { "False",
      "None",
      "True",
      "and",
      "as",
      "assert",
      "async",
      "await",
      "break",
      "class",
      "continue",
      "def",
      "del",
      "elif",
      "else",
      "except",
      "finally",
      "for",
      "from",
      "global",
      "if",
      "import",
      "in",
      "is",
      "lambda",
      "nonlocal",
      "not",
      "or",
      "pass",
      "raise",
      "return",
      "try",
      "while",
      "with",
      "yield"};
  varNameGenerator = util::NameGenerator(keywords);
}

void DacePrinter::doIndent() {
  for (int i=0; i<indent; i++)
    stream << "    ";
}

void DacePrinter::printBinOp(Expr a, Expr b, string op, Precedence precedence) {
  bool parenthesize = needsParentheses(precedence);
  if (parenthesize) {
    stream << "(";
  }
  parentPrecedence = precedence;
  a.accept(this);
  stream << " " << op << " ";
  parentPrecedence = precedence;
  b.accept(this);
  if (parenthesize) {
    stream << ")";
  }
}

bool DacePrinter::needsParentheses(Precedence precedence) {
  // Add parentheses if required by C operator precedence or for Boolean
  // expressions of form `a || (b && c)` (to avoid C compiler warnings)
  return (precedence >= parentPrecedence ||
            (precedence == Precedence::LAND &&
              parentPrecedence == Precedence::LOR));
}


std::string DacePrinter::keywordString(std::string keyword) {
  if (color) {
    return magenta + keyword + nc;
  }
  else {
    return keyword;
  }
}

std::string DacePrinter::commentString(std::string comment) {
  if (color) {
    return green + "\"\"\" " + comment + " \"\"\"" + nc;
  }
  else {
    return "\"\"\" " + comment + " \"\"\"";
  }
}



}}
