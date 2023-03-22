#include <cstdlib>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <map>
#include <memory>

#include "taco.h"

#include "taco/error.h"
#include "taco/parser/lexer.h"
#include "taco/parser/parser.h"
#include "taco/parser/schedule_parser.h"
#include "taco/storage/storage.h"
#include "taco/ir/ir.h"
#include "taco/ir/ir_printer.h"
#include "taco/index_notation/kernel.h"
#include "lower/iteration_graph.h"
#include "taco/lower/lower.h"
#include "taco/codegen/module.h"
#include "codegen/codegen_c.h"
#include "codegen/codegen_cuda.h"
#include "codegen/codegen.h"
#include "taco/util/strings.h"
#include "taco/util/files.h"
#include "taco/util/timers.h"
#include "taco/util/fill.h"
#include "taco/util/env.h"
#include "taco/util/collections.h"
#include "taco/cuda.h"
#include "taco/index_notation/transformations.h"
#include "taco/index_notation/index_notation_visitor.h"
#include "taco/index_notation/index_notation_nodes.h"
#include "taco/version.h"

using namespace std;
using namespace taco;

static void printFlag(string flag, string text) {
  const size_t descriptionStart = 30;
  const size_t columnEnd        = 80;
  string flagString = "  -" + flag +
                      util::repeat(" ",descriptionStart-(flag.size()+3));
  cout << flagString;
  size_t column = flagString.size();
  vector<string> words = util::split(text, " ");
  for (auto& word : words) {
    if (column + word.size()+1 >= columnEnd) {
      cout << endl << util::repeat(" ", descriptionStart);
      column = descriptionStart;
    }
    column += word.size()+1;
    cout << word << " ";
  }
  cout << endl;
}

static void printUsageInfo() {
  cout << "Usage: dace_taco <index expression> [options]" << endl;
  cout << endl;
  cout << "Examples:" << endl;
  cout << "  dace_taco \"a(i) = b(i) + c(i)\"                            # Dense vector add" << endl;
  cout << "  dace_taco \"a(i) = b(i) + c(i)\" -f=b:s -f=c:s -f=a:s       # Sparse vector add" << endl;
  cout << "  dace_taco \"a(i) = B(i,j) * c(j)\" -f=B:ds                  # SpMV" << endl;
  cout << "  dace_taco \"A(i,l) = B(i,j,k) * C(j,l) * D(k,l)\" -f=B:sss  # MTTKRP" << endl;
  cout << endl;
  cout << "Options:" << endl;
  printFlag("f=<tensor>:<format>",
            "Specify the format of a tensor in the expression. Formats are "
            "specified per dimension using d (dense), s (sparse), "
            "u (sparse, not unique), q (singleton), c (singleton, not unique), "
            "or p (singleton, padded). All formats default to dense. "
            "The ordering of modes can also be optionally specified as a "
            "comma-delimited list of modes in the order they should be stored. "
            "Examples: A:ds (i.e., CSR), B:ds:1,0 (i.e., CSC), c:d (i.e., "
            "dense vector), D:sss (i.e., CSF).");
  cout << endl;
  printFlag("s=\"<command>(<params>)\"",
            "Specify a scheduling command to apply to the generated code. "
            "Parameters take the form of a comma-delimited list. See "
            "-help=scheduling for a list of scheduling commands. "
            "Examples: split(i,i0,i1,16), precompute(A(i,j)*x(j),i,i).");
  cout << endl;
  printFlag("c",
            "Generate compute kernel that simultaneously does assembly.");
  cout << endl;
  printFlag("O=<directory path>",
            "Write all files to a directory. "
            "(defaults to $TMPDIR)");
  cout << endl;
  printFlag("write-compute",
            "Write the compute kernel to a file.");
  cout << endl;
  printFlag("write-assemble",
            "Write the assembly kernel to a file.");
  cout << endl;
  printFlag("write-concrete",
            "Write the concrete index notation of this expression.");
  cout << endl;
  printFlag("write-iteration-graph",
            "Write the iteration graph of this expression in the dot format.");
  cout << endl;
  printFlag("prefix", "Specify a prefix for generated function names");
  cout << endl;

}

static void printSchedulingHelp() {
    cout << "Scheduling commands modify the execution of the index expression." << endl;
    cout << "The '-s' parameter specifies one or more scheduling commands." << endl;
    cout << "Schedules are additive; more commands can be passed by separating" << endl;
    cout << "them with commas, or passing multiple '-s' parameters." << endl;
    cout << endl;
    cout << "Examples:" << endl;
    cout << "  -s=\"precompute(A(i,j)*x(j),i,i)\"" << endl;
    cout << "  -s=\"split(i,i0,i1,32),parallelize(i0,CPUThread,NoRaces)\"" << endl;
    cout << endl;
    cout << "See http://tensor-compiler.org/docs/scheduling/index.html for more examples." << endl;
    cout << endl;
    cout << "Commands:" << endl;
    printFlag("s=pos(i, ipos, tensor)", "Takes in an index variable `i` "
              "that iterates over the coordinate space of `tensor` and replaces "
              "it with a derived index variable `ipos` that iterates over the "
              "same iteration range, but with respect to the the position space. "
              "The `pos` transformation is not valid for dense level formats.");
    cout << endl;
    printFlag("s=fuse(i, j, f)", "Takes in two index variables `i` and `j`, where "
              "`j` is directly nested under `i`, and collapses them into a fused "
              "index variable `f` that iterates over the product of the "
              "coordinates `i` and `j`.");
    cout << endl;
    printFlag("s=split(i, i0, i1, factor)", "Splits (strip-mines) an index "
              "variable `i` into two nested index variables `i0` and `i1`. The "
              "size of the inner index variable `i1` is then held constant at "
              "`factor`, which must be a positive integer.");
    cout << endl;
    printFlag("s=precompute(expr, i, iw)", "Leverages scratchpad memories and "
              "reorders computations to increase locality.  Given a subexpression "
              "`expr` to precompute, an index variable `i` to precompute over, "
              "and an index variable `iw` (which can be the same or different as "
              "`i`) to precompute with, the precomputed results are stored in a "
              "temporary tensor variable.");
    cout << endl;
    printFlag("s=reorder(i1, i2, ...)", "Takes in a new ordering for a "
              "set of index variables in the expression that are directly nested "
              "in the iteration order.  The indexes are ordered from outermost "
              "to innermost.");
    cout << endl;
    printFlag("s=bound(i, ib, b, type)", "Replaces an index variable `i` "
              "with an index variable `ib` that obeys a compile-time constraint "
              "on its iteration space, incorporating knowledge about the size or "
              "structured sparsity pattern of the corresponding input. The "
              "meaning of `b` depends on the `type`. Possible bound types are: "
              "MinExact, MinConstraint, MaxExact, MaxConstraint.");
    cout << endl;
    printFlag("s=unroll(index, factor)", "Unrolls the loop corresponding to an "
              "index variable `i` by `factor` number of iterations, where "
              "`factor` is a positive integer.");
    cout << endl;
    printFlag("s=parallelize(i, u, strat)", "tags an index variable `i` for "
              "parallel execution on hardware type `u`. Data races are handled by "
              "an output race strategy `strat`. Since the other transformations "
              "expect serial code, parallelize must come last in a series of "
              "transformations.  Possible parallel hardware units are: "
              "NotParallel, GPUBlock, GPUWarp, GPUThread, CPUThread, CPUVector. "
              "Possible output race strategies are: "
              "IgnoreRaces, NoRaces, Atomics, Temporary, ParallelReduction.");
}

static void printVersionInfo() {
  string gitsuffix("");
  if(strlen(TACO_VERSION_GIT_SHORTHASH) > 0) {
    gitsuffix = string("+git " TACO_VERSION_GIT_SHORTHASH);
  }
  cout << "TACO version: " << TACO_VERSION_MAJOR << "." << TACO_VERSION_MINOR << gitsuffix << endl;
  if(TACO_FEATURE_OPENMP)
    cout << "Built with OpenMP support." << endl;
  if(TACO_FEATURE_PYTHON)
    cout << "Built with Python support." << endl;
  if(TACO_FEATURE_CUDA)
    cout << "Built with CUDA support." << endl;
  cout << endl;
  cout << "Built on: " << TACO_BUILD_DATE << endl;
  cout << "CMake build type: " << TACO_BUILD_TYPE << endl;
  cout << "Built with compiler: " << TACO_BUILD_COMPILER_ID << " C++ version " << TACO_BUILD_COMPILER_VERSION << endl;
}

static int reportError(string errorMessage, int errorCode) {
  cerr << "Error: " << errorMessage << endl << endl;
  printUsageInfo();
  return errorCode;
}

static void printCommandLine(ostream& os, int argc, char* argv[]) {
  taco_iassert(argc > 0);
  os << argv[0];
  if (argc > 1) {
    os << " \"" << argv[1] << "\"";
  }
  for (int i = 2; i < argc; i++) {
    os << " ";
    std::string arg = argv[i];
    if (arg.rfind("-s=", 0) == 0) {
      arg.replace(0, 3, "-s=\"");
      arg += "\"";
    }
    os << arg;
  }
}

static bool setSchedulingCommands(vector<vector<string>> scheduleCommands, parser::Parser& parser, IndexStmt& stmt) {
  auto findVar = [&stmt](string name) {
    ProvenanceGraph graph(stmt);
    for (auto v : graph.getAllIndexVars()) {
      if (v.getName() == name) {
        return v;
      }
    }

    taco_uassert(0) << "Index variable '" << name << "' not defined in statement " << stmt;
    abort(); // to silence a warning: control reaches end of non-void function
  };

  bool isGPU = false;

  for(vector<string> scheduleCommand : scheduleCommands) {
    string command = scheduleCommand[0];
    scheduleCommand.erase(scheduleCommand.begin());

    if (command == "pos") {
      taco_uassert(scheduleCommand.size() == 3) << "'pos' scheduling directive takes 3 parameters: pos(i, ipos, tensor)";
      string i, ipos, tensor;
      i      = scheduleCommand[0];
      ipos   = scheduleCommand[1];
      tensor = scheduleCommand[2];

      for (auto a : getArgumentAccesses(stmt)) {
        if (a.getTensorVar().getName() == tensor) {
          IndexVar derived(ipos);
          stmt = stmt.pos(findVar(i), derived, a);
          goto end;
        }
      }

    } else if (command == "fuse") {
      taco_uassert(scheduleCommand.size() == 3) << "'fuse' scheduling directive takes 3 parameters: fuse(i, j, f)";
      string i, j, f;
      i = scheduleCommand[0];
      j = scheduleCommand[1];
      f = scheduleCommand[2];

      IndexVar fused(f);
      stmt = stmt.fuse(findVar(i), findVar(j), fused);

    } else if (command == "split") {
      taco_uassert(scheduleCommand.size() == 4)
          << "'split' scheduling directive takes 4 parameters: split(i, i1, i2, splitFactor)";
      string i, i1, i2;
      size_t splitFactor;
      i = scheduleCommand[0];
      i1 = scheduleCommand[1];
      i2 = scheduleCommand[2];
      taco_uassert(sscanf(scheduleCommand[3].c_str(), "%zu", &splitFactor) == 1)
          << "failed to parse fourth parameter to `split` directive as a size_t";

      IndexVar split1(i1);
      IndexVar split2(i2);
      stmt = stmt.split(findVar(i), split1, split2, splitFactor);
    } else if (command == "divide") {
      taco_uassert(scheduleCommand.size() == 4)
          << "'divide' scheduling directive takes 4 parameters: divide(i, i1, i2, divFactor)";
      string i, i1, i2;
      i = scheduleCommand[0];
      i1 = scheduleCommand[1];
      i2 = scheduleCommand[2];

      size_t divideFactor;
      taco_uassert(sscanf(scheduleCommand[3].c_str(), "%zu", &divideFactor) == 1)
          << "failed to parse fourth parameter to `divide` directive as a size_t";

      IndexVar divide1(i1);
      IndexVar divide2(i2);
      stmt = stmt.divide(findVar(i), divide1, divide2, divideFactor);
    } else if (command == "precompute") {
      string exprStr, i, iw, name;
      vector<string> i_vars, iw_vars;

      taco_uassert(scheduleCommand.size() == 3 || scheduleCommand.size() == 4)
        << "'precompute' scheduling directive takes 3 or 4 parameters: "
        << "precompute(expr, i, iw [, workspace_name]) or precompute(expr, {i_vars}, "
           "{iw_vars} [, workspace_name])" << scheduleCommand.size();

      exprStr = scheduleCommand[0];
//      i       = scheduleCommand[1];
//      iw      = scheduleCommand[2];
      i_vars  = parser::varListParser(scheduleCommand[1]);
      iw_vars = parser::varListParser(scheduleCommand[2]);

      if (scheduleCommand.size() == 4)
        name  = scheduleCommand[3];
      else
        name  = "workspace";

      vector<IndexVar> origs;
      vector<IndexVar> pres;
      for (auto& i : i_vars) {
        origs.push_back(findVar(i));
      }
      for (auto& iw : iw_vars) {
        try {
          pres.push_back(findVar(iw));
        } catch (TacoException &e) {
          pres.push_back(IndexVar(iw));
        }
      }

      struct GetExpr : public IndexNotationVisitor {
        using IndexNotationVisitor::visit;

        string exprStr;
        IndexExpr expr;

        void setExprStr(string input) {
          exprStr = input;
          exprStr.erase(remove(exprStr.begin(), exprStr.end(), ' '), exprStr.end());
        }

        string toString(IndexExpr e) {
          stringstream tempStream;
          tempStream << e;
          string tempStr = tempStream.str();
          tempStr.erase(remove(tempStr.begin(), tempStr.end(), ' '), tempStr.end());
          return tempStr;
        }

        void visit(const AccessNode* node) {
          IndexExpr currentExpr(node);
          if (toString(currentExpr) == exprStr) {
            expr = currentExpr;
          }
          else {
            IndexNotationVisitor::visit(node);
          }
        }

        void visit(const UnaryExprNode* node) {
          IndexExpr currentExpr(node);
          if (toString(currentExpr) == exprStr) {
            expr = currentExpr;
          }
          else {
            IndexNotationVisitor::visit(node);
          }
        }

        void visit(const BinaryExprNode* node) {
          IndexExpr currentExpr(node);
          if (toString(currentExpr) == exprStr) {
            expr = currentExpr;
          }
          else {
            IndexNotationVisitor::visit(node);
          }
        }
      };

      GetExpr visitor;
      visitor.setExprStr(exprStr);
      stmt.accept(&visitor);

      vector<Dimension> dims;
      auto domains = stmt.getIndexVarDomains();
      for (auto& orig : origs) {
        auto it = domains.find(orig);
        if (it != domains.end()) {
          dims.push_back(it->second);
        } else {
          dims.push_back(Dimension(orig));
        }
      }

      std::vector<ModeFormatPack> modeFormatPacks(dims.size(), Dense);
      Format format(modeFormatPacks);
      TensorVar workspace(name, Type(Float64, dims), format);

      stmt = stmt.precompute(visitor.expr, origs, pres, workspace);

    } else if (command == "reorder") {
      taco_uassert(scheduleCommand.size() > 1) << "'reorder' scheduling directive needs at least 2 parameters: reorder(outermost, ..., innermost)";

      vector<IndexVar> reorderedVars;
      for (string var : scheduleCommand) {
        reorderedVars.push_back(findVar(var));
      }

      stmt = stmt.reorder(reorderedVars);

    } else if (command == "mergeby") {
      taco_uassert(scheduleCommand.size() == 2) << "'mergeby' scheduling directive takes 2 parameters: mergeby(i, strategy)";
      string i, strat;
      MergeStrategy strategy;

      i = scheduleCommand[0];
      strat = scheduleCommand[1];
      if (strat == "TwoFinger") {
        strategy = MergeStrategy::TwoFinger;
      } else if (strat == "Gallop") {
        strategy = MergeStrategy::Gallop;
      } else {
        taco_uerror << "Merge strategy not defined.";
        goto end;
      }

      stmt = stmt.mergeby(findVar(i), strategy);

    } else if (command == "bound") {
      taco_uassert(scheduleCommand.size() == 4) << "'bound' scheduling directive takes 4 parameters: bound(i, i1, bound, type)";
      string i, i1, type;
      size_t bound;
      i  = scheduleCommand[0];
      i1 = scheduleCommand[1];
      taco_uassert(sscanf(scheduleCommand[2].c_str(), "%zu", &bound) == 1) << "failed to parse third parameter to `bound` directive as a size_t";
      type = scheduleCommand[3];

      BoundType bound_type;
      if (type == "MinExact") {
        bound_type = BoundType::MinExact;
      } else if (type == "MinConstraint") {
        bound_type = BoundType::MinConstraint;
      } else if (type == "MaxExact") {
        bound_type = BoundType::MaxExact;
      } else if (type == "MaxConstraint") {
        bound_type = BoundType::MaxConstraint;
      } else {
        taco_uerror << "Bound type not defined.";
        goto end;
      }

      IndexVar bound1(i1);
      stmt = stmt.bound(findVar(i), bound1, bound, bound_type);

    } else if (command == "unroll") {
      taco_uassert(scheduleCommand.size() == 2) << "'unroll' scheduling directive takes 2 parameters: unroll(i, unrollFactor)";
      string i;
      size_t unrollFactor;
      i  = scheduleCommand[0];
      taco_uassert(sscanf(scheduleCommand[1].c_str(), "%zu", &unrollFactor) == 1) << "failed to parse second parameter to `unroll` directive as a size_t";

      stmt = stmt.unroll(findVar(i), unrollFactor);

    } else if (command == "parallelize") {
      string i, unit, strategy;
      taco_uassert(scheduleCommand.size() == 3) << "'parallelize' scheduling directive takes 3 parameters: parallelize(i, unit, strategy)";
      i        = scheduleCommand[0];
      unit     = scheduleCommand[1];
      strategy = scheduleCommand[2];

      ParallelUnit parallel_unit;
      if (unit == "NotParallel") {
        parallel_unit = ParallelUnit::NotParallel;
      } else if (unit == "GPUBlock") {
        parallel_unit = ParallelUnit::GPUBlock;
        isGPU = true;
      } else if (unit == "GPUWarp") {
        parallel_unit = ParallelUnit::GPUWarp;
        isGPU = true;
      } else if (unit == "GPUThread") {
        parallel_unit = ParallelUnit::GPUThread;
        isGPU = true;
      } else if (unit == "CPUThread") {
        parallel_unit = ParallelUnit::CPUThread;
      } else if (unit == "CPUVector") {
        parallel_unit = ParallelUnit::CPUVector;
      } else {
        taco_uerror << "Parallel hardware not defined.";
        goto end;
      }

      OutputRaceStrategy output_race_strategy;
      if (strategy == "IgnoreRaces") {
        output_race_strategy = OutputRaceStrategy::IgnoreRaces;
      } else if (strategy == "NoRaces") {
        output_race_strategy = OutputRaceStrategy::NoRaces;
      } else if (strategy == "Atomics") {
        output_race_strategy = OutputRaceStrategy::Atomics;
      } else if (strategy == "Temporary") {
        output_race_strategy = OutputRaceStrategy::Temporary;
      } else if (strategy == "ParallelReduction") {
        output_race_strategy = OutputRaceStrategy::ParallelReduction;
      } else {
        taco_uerror << "Race strategy not defined.";
        goto end;
      }

      stmt = stmt.parallelize(findVar(i), parallel_unit, output_race_strategy);

    } else if (command == "assemble") {
      taco_uassert(scheduleCommand.size() == 2 || scheduleCommand.size() == 3) 
          << "'assemble' scheduling directive takes 2 or 3 parameters: "
          << "assemble(tensor, strategy [, separately_schedulable])";

      string tensor = scheduleCommand[0];
      string strategy = scheduleCommand[1];
      string schedulable = "false";
      if (scheduleCommand.size() == 3) {
        schedulable = scheduleCommand[2];
      }

      TensorVar result;
      for (auto a : getResultAccesses(stmt).first) {
        if (a.getTensorVar().getName() == tensor) {
          result = a.getTensorVar();
          break;
        }
      }
      taco_uassert(result.defined()) << "Unable to find result tensor '"
                                     << tensor << "'";

      AssembleStrategy assemble_strategy;
      if (strategy == "Append") {
        assemble_strategy = AssembleStrategy::Append;
      } else if (strategy == "Insert") {
        assemble_strategy = AssembleStrategy::Insert;
      } else {
        taco_uerror << "Assemble strategy not defined.";
        goto end;
      }

      bool separately_schedulable;
      if (schedulable == "true") {
        separately_schedulable = true;
      } else if (schedulable == "false") {
        separately_schedulable = false;
      } else {
        taco_uerror << "Incorrectly specified whether computation of result "
                    << "statistics should be separately schedulable.";
        goto end;
      }

      stmt = stmt.assemble(result, assemble_strategy, separately_schedulable);

    } else {
      taco_uerror << "Unknown scheduling function \"" << command << "\"";
      break;
    }

    end:;
  }

  return isGPU;
}

int main(int argc, char* argv[]) {
  if (argc < 2) {
    printUsageInfo();
    return 0;
  }

  bool computeWithAssemble = false;

  bool writeCompute        = false;
  bool writeAssemble       = false;
  bool writeConcrete       = false;
  bool writeIterationGraph = false;

  bool setSchedule         = false;

  ParallelSchedule sched = ParallelSchedule::Static;
  int chunkSize = 0;
  int nthreads = 0;
  string prefix = "";

  string indexVarName = "";

  string exprStr;
  map<string,Format> formats;
  map<string,std::vector<int>> tensorsDimensions;
  map<string,Datatype> dataTypes;
  map<string,taco::util::FillMethod> tensorsFill;
  map<string,string> inputFilenames;
  map<string,string> outputFilenames;
  string outputDirectory;
  string writeComputeFilename;
  string writeAssembleFilename;
  string writeKernelFilename;
  string writeTimeFilename;
  vector<string> declaredTensors;

  vector<string> kernelFilenames;

  vector<vector<string>> scheduleCommands;
  
  for (int i = 1; i < argc; i++) {
    string arg = argv[i];
    if(arg.rfind("--", 0) == 0) {
      // treat leading "--" as if it were "-"
      arg = string(argv[i]+1);
    }
    vector<string> argparts = util::split(arg, "=");
    if (argparts.size() > 2) {
      return reportError("Too many '\"' signs in argument", 5);
    }
    string argName = argparts[0];
    string argValue;
    if (argparts.size() == 2)
      argValue = argparts[1];

    if ("-help" == argName) {
        if(argValue == "scheduling") {
            printSchedulingHelp();
        } else {
            printUsageInfo();
        }
        return 0;
    }
    if ("-version" == argName) {
        printVersionInfo();
        return 0;
    }
    else if ("-f" == argName) {
      vector<string> descriptor = util::split(argValue, ":");
      if (descriptor.size() < 2 || descriptor.size() > 4) {
        return reportError("Incorrect format descriptor", 4);
      }
      string tensorName = descriptor[0];
      string formatString = descriptor[1];
      std::vector<ModeFormat> modeTypes;
      std::vector<ModeFormatPack> modeTypePacks;
      std::vector<int> modeOrdering;
      for (int i = 0; i < (int)formatString.size(); i++) {
        switch (formatString[i]) {
          case 'd':
            modeTypes.push_back(ModeFormat::Dense);
            break;
          case 's':
            modeTypes.push_back(ModeFormat::Sparse);
            break;
          case 'u':
            modeTypes.push_back(ModeFormat::Sparse(ModeFormat::NOT_UNIQUE));
            break;
          case 'z':
            modeTypes.push_back(ModeFormat::Sparse(ModeFormat::ZEROLESS));
            break;
          case 'c':
            modeTypes.push_back(ModeFormat::Singleton(ModeFormat::NOT_UNIQUE));
            break;
          case 'q':
            modeTypes.push_back(ModeFormat::Singleton);
            break;
          case 'p':
            modeTypes.push_back(ModeFormat::Singleton(ModeFormat::PADDED));
            break;
          default:
            return reportError("Incorrect format descriptor", 3);
            break;
        }
        modeOrdering.push_back(i);
      }
      if (descriptor.size() > 2) {
        std::vector<std::string> modes = util::split(descriptor[2], ",");
        modeOrdering.clear();
        for (const auto& mode : modes) {
          modeOrdering.push_back(std::stoi(mode));
        }
      }
      if (descriptor.size() > 3) {
        std::vector<std::string> packBoundStrs = util::split(descriptor[3], ",");
        std::vector<int> packBounds(packBoundStrs.size());
        for (int i = 0; i < (int)packBounds.size(); ++i) {
          packBounds[i] = std::stoi(packBoundStrs[i]);
        }
        int pack = 0;
        std::vector<ModeFormat> modeTypesInPack;
        for (int i = 0; i < (int)modeTypes.size(); ++i) {
          if (i == packBounds[pack]) {
            modeTypePacks.push_back(modeTypesInPack);
            modeTypesInPack.clear();
            ++pack;
          }
          modeTypesInPack.push_back(modeTypes[i]);
        }
        modeTypePacks.push_back(modeTypesInPack);
      } else {
        for (const auto& modeType : modeTypes) {
          modeTypePacks.push_back(modeType);
        }
      }
      formats.insert({tensorName, Format(modeTypePacks, modeOrdering)});
    }
    else if ("-c" == argName) {
      computeWithAssemble = true;
    }
    else if ("-O" == argName) {
      if (util::split(argValue, ":").size() > 1) {
        return reportError("Incorrect -O usage", 3);
      }
      outputDirectory = (argValue != "") ? argValue : util::getTmpdir();
    }
    else if ("-write-compute" == argName) {
      writeCompute = true;
    }
    else if ("-write-assemble" == argName) {
      writeAssemble = true;
    }
    else if ("-write-concrete" == argName) {
      writeConcrete = true;
    }
    else if ("-write-iteration-graph" == argName) {
      writeIterationGraph = true;
    }
    else if ("-s" == argName) {
      setSchedule = true;
      vector<vector<string>> parsed = parser::ScheduleParser(argValue);

      taco_uassert(parsed.size() > 0) << "-s parameter got no scheduling directives?";
      for(vector<string> directive : parsed)
        scheduleCommands.push_back(directive);
    }
    else if ("-prefix" == argName) {
      prefix = argValue;
    }
    else {
      if (exprStr.size() != 0) {
        printUsageInfo();
        return 2;
      }
      exprStr = argv[i];
    }
  }

  if (exprStr == "") {
    return 0;
  }
  map<string,TensorBase> loadedTensors;
  TensorBase tensor;
  parser::Parser parser(exprStr, formats, dataTypes, tensorsDimensions, loadedTensors, 42);
  try {
    parser.parse();
    tensor = parser.getResultTensor();
  } catch (parser::ParseError& e) {
    return reportError(e.getMessage(), 6);
  }

  ir::Stmt assemble;
  ir::Stmt compute;

  taco_set_parallel_schedule(sched, chunkSize);
  taco_set_num_threads(nthreads);

  IndexStmt stmt =
      makeConcreteNotation(makeReductionNotation(tensor.getAssignment()));
  stmt = reorderLoopsTopologically(stmt);

  if (setSchedule) {
    setSchedulingCommands(scheduleCommands, parser, stmt);
  }
  else {
    stmt = insertTemporaries(stmt);
    stmt = parallelizeOuterLoop(stmt);
  }
  set_CUDA_codegen_enabled(false);

  stmt = scalarPromote(stmt);
  
  if (writeConcrete) {
    std::string filePath = outputDirectory + '/' + prefix + "concrete.txt";
    
    std::ofstream filestream;
    filestream.open(filePath,
                    std::ofstream::out|std::ofstream::trunc);
    filestream << stmt << endl;
    filestream.close();
  }


  compute = lower(stmt, prefix+"compute",  computeWithAssemble, true);
  assemble = lower(stmt, prefix+"assemble", true, false);


  IterationGraph iterationGraph;
  if (writeIterationGraph) {
    iterationGraph = IterationGraph::make(tensor.getAssignment());
    std::string filePath = outputDirectory + '/' + prefix + "iterationGraph.dot";
    
    std::ofstream filestream;
    filestream.open(filePath,
                    std::ofstream::out|std::ofstream::trunc);
    iterationGraph.printAsDot(filestream);
    filestream.close();
  }

  if (writeCompute) {
    std::string filePath = outputDirectory + '/' + prefix + "compute.py";
    
    std::ofstream filestream;
    filestream.open(filePath,
                    std::ofstream::out|std::ofstream::trunc);
    ir::DacePrinter dprinter = ir::DacePrinter(filestream, false, true);
    dprinter.print(compute);
    filestream << endl << "sdfg = " << prefix << "compute.to_sdfg()" << endl;
    filestream << "sdfg.save(\"" << outputDirectory << "/" << prefix << "compute.sdfg\")" << endl;
    filestream.close();    
  }

  if (writeAssemble) {
    std::string filePath = outputDirectory + '/' + prefix + "assemble.py";
    
    std::ofstream filestream;
    filestream.open(filePath,
                    std::ofstream::out|std::ofstream::trunc);
    ir::DacePrinter dprinter = ir::DacePrinter(filestream, false, true);
    dprinter.print(assemble);
    filestream << endl << "sdfg = " << prefix << "assemble.to_sdfg()" << endl;
    filestream << "sdfg.save(\"" << outputDirectory << "/" << prefix << "assemble.sdfg\")" << endl;
    filestream.close();
  }
}