"use strict";
var __awaiter = (this && this.__awaiter) || function (thisArg, _arguments, P, generator) {
    return new (P || (P = Promise))(function (resolve, reject) {
        function fulfilled(value) { try { step(generator.next(value)); } catch (e) { reject(e); } }
        function rejected(value) { try { step(generator["throw"](value)); } catch (e) { reject(e); } }
        function step(result) { result.done ? resolve(result.value) : new P(function (resolve) { resolve(result.value); }).then(fulfilled, rejected); }
        step((generator = generator.apply(thisArg, _arguments || [])).next());
    });
};
var __generator = (this && this.__generator) || function (thisArg, body) {
    var _ = { label: 0, sent: function() { if (t[0] & 1) throw t[1]; return t[1]; }, trys: [], ops: [] }, f, y, t, g;
    return g = { next: verb(0), "throw": verb(1), "return": verb(2) }, typeof Symbol === "function" && (g[Symbol.iterator] = function() { return this; }), g;
    function verb(n) { return function (v) { return step([n, v]); }; }
    function step(op) {
        if (f) throw new TypeError("Generator is already executing.");
        while (_) try {
            if (f = 1, y && (t = y[op[0] & 2 ? "return" : op[0] ? "throw" : "next"]) && !(t = t.call(y, op[1])).done) return t;
            if (y = 0, t) op = [0, t.value];
            switch (op[0]) {
                case 0: case 1: t = op; break;
                case 4: _.label++; return { value: op[1], done: false };
                case 5: _.label++; y = op[1]; op = [0]; continue;
                case 7: op = _.ops.pop(); _.trys.pop(); continue;
                default:
                    if (!(t = _.trys, t = t.length > 0 && t[t.length - 1]) && (op[0] === 6 || op[0] === 2)) { _ = 0; continue; }
                    if (op[0] === 3 && (!t || (op[1] > t[0] && op[1] < t[3]))) { _.label = op[1]; break; }
                    if (op[0] === 6 && _.label < t[1]) { _.label = t[1]; t = op; break; }
                    if (t && _.label < t[2]) { _.label = t[2]; _.ops.push(op); break; }
                    if (t[2]) _.ops.pop();
                    _.trys.pop(); continue;
            }
            op = body.call(thisArg, _);
        } catch (e) { op = [6, e]; y = 0; } finally { f = t = 0; }
        if (op[0] & 5) throw op[1]; return { value: op[0] ? op[1] : void 0, done: true };
    }
};
var _this = this;
Object.defineProperty(exports, "__esModule", { value: true });
var tfc = require("@tensorflow/tfjs-core");
var tfjs_core_1 = require("@tensorflow/tfjs-core");
var test_util_1 = require("@tensorflow/tfjs-core/dist/test_util");
var execution_context_1 = require("../../executor/execution_context");
var tensor_array_1 = require("../../executor/tensor_array");
var control_executor_1 = require("./control_executor");
var test_helper_1 = require("./test_helper");
describe('control', function () {
    var node;
    var input1 = [tfc.scalar(1, 'int32')];
    var context = new execution_context_1.ExecutionContext({}, {});
    beforeEach(function () {
        node = {
            name: 'test',
            op: '',
            category: 'control',
            inputNames: ['pred', 'input1'],
            inputs: [],
            params: { x: test_helper_1.createTensorAttr(0) },
            children: []
        };
    });
    describe('executeOp', function () {
        describe('switch', function () {
            it('should set the output condition is true', function () { return __awaiter(_this, void 0, void 0, function () {
                var pred, _a;
                return __generator(this, function (_b) {
                    switch (_b.label) {
                        case 0:
                            node.op = 'switch';
                            node.params['pred'] = test_helper_1.createTensorAttr(0);
                            node.params['data'] = test_helper_1.createTensorAttr(1);
                            pred = [tfc.scalar(true)];
                            _a = expect;
                            return [4, control_executor_1.executeOp(node, { pred: pred, input1: input1 }, context)];
                        case 1:
                            _a.apply(void 0, [_b.sent()]).toEqual([
                                undefined, input1[0]
                            ]);
                            return [2];
                    }
                });
            }); });
            it('should set the output condition is false', function () { return __awaiter(_this, void 0, void 0, function () {
                var pred, _a;
                return __generator(this, function (_b) {
                    switch (_b.label) {
                        case 0:
                            node.op = 'switch';
                            node.params['pred'] = test_helper_1.createTensorAttr(0);
                            node.params['data'] = test_helper_1.createTensorAttr(1);
                            pred = [tfc.scalar(false)];
                            _a = expect;
                            return [4, control_executor_1.executeOp(node, { pred: pred, input1: input1 }, context)];
                        case 1:
                            _a.apply(void 0, [_b.sent()]).toEqual([
                                input1[0], undefined
                            ]);
                            return [2];
                    }
                });
            }); });
        });
        describe('merge', function () {
            it('should return the first available input', function () { return __awaiter(_this, void 0, void 0, function () {
                var pred, _a, _b;
                return __generator(this, function (_c) {
                    switch (_c.label) {
                        case 0:
                            node.op = 'merge';
                            pred = [tfc.scalar(true)];
                            _a = expect;
                            return [4, control_executor_1.executeOp(node, { pred: undefined, input1: input1 }, context)];
                        case 1:
                            _a.apply(void 0, [_c.sent()])
                                .toEqual(input1);
                            _b = expect;
                            return [4, control_executor_1.executeOp(node, { pred: pred, input1: undefined }, context)];
                        case 2:
                            _b.apply(void 0, [_c.sent()])
                                .toEqual(pred);
                            return [2];
                    }
                });
            }); });
            it('should return undefined if no inputs are available', function () { return __awaiter(_this, void 0, void 0, function () {
                var _a;
                return __generator(this, function (_b) {
                    switch (_b.label) {
                        case 0:
                            node.op = 'merge';
                            _a = expect;
                            return [4, control_executor_1.executeOp(node, { pred: undefined, input1: undefined }, context)];
                        case 1:
                            _a.apply(void 0, [_b.sent()])
                                .toEqual(undefined);
                            return [2];
                    }
                });
            }); });
        });
        describe('enter', function () {
            it('should call enterFrame on context', function () { return __awaiter(_this, void 0, void 0, function () {
                var _a;
                return __generator(this, function (_b) {
                    switch (_b.label) {
                        case 0:
                            spyOn(context, 'enterFrame');
                            node.op = 'enter';
                            node.params['tensor'] = test_helper_1.createTensorAttr(0);
                            node.inputNames = ['input1'];
                            _a = expect;
                            return [4, control_executor_1.executeOp(node, { input1: input1 }, context)];
                        case 1:
                            _a.apply(void 0, [_b.sent()]).toEqual(input1);
                            expect(context.enterFrame).toHaveBeenCalled();
                            return [2];
                    }
                });
            }); });
        });
        describe('exit', function () {
            it('should call existFrame on context', function () { return __awaiter(_this, void 0, void 0, function () {
                var _a;
                return __generator(this, function (_b) {
                    switch (_b.label) {
                        case 0:
                            spyOn(context, 'exitFrame');
                            node.op = 'exit';
                            node.params['tensor'] = test_helper_1.createTensorAttr(0);
                            node.inputNames = ['input1'];
                            _a = expect;
                            return [4, control_executor_1.executeOp(node, { input1: input1 }, context)];
                        case 1:
                            _a.apply(void 0, [_b.sent()]).toEqual(input1);
                            expect(context.exitFrame).toHaveBeenCalled();
                            return [2];
                    }
                });
            }); });
        });
        describe('nextIteration', function () {
            it('should call nextIteration on context', function () { return __awaiter(_this, void 0, void 0, function () {
                var _a;
                return __generator(this, function (_b) {
                    switch (_b.label) {
                        case 0:
                            spyOn(context, 'nextIteration');
                            node.op = 'nextIteration';
                            node.params['tensor'] = test_helper_1.createTensorAttr(0);
                            node.inputNames = ['input1'];
                            _a = expect;
                            return [4, control_executor_1.executeOp(node, { input1: input1 }, context)];
                        case 1:
                            _a.apply(void 0, [_b.sent()]).toEqual(input1);
                            expect(context.nextIteration).toHaveBeenCalled();
                            return [2];
                    }
                });
            }); });
        });
        describe('tensorArray', function () {
            it('should create new tensor on the context', function () { return __awaiter(_this, void 0, void 0, function () {
                var tensorId;
                return __generator(this, function (_a) {
                    switch (_a.label) {
                        case 0:
                            node.op = 'tensorArray';
                            node.params['name'] = test_helper_1.createStrAttr('');
                            node.params['dtype'] = test_helper_1.createDtypeAttr('int32');
                            node.params['elementShape'] = test_helper_1.createNumericArrayAttr([10, 10]);
                            node.params['dynamicSize'] = test_helper_1.createBoolAttr(false);
                            node.params['clearAfterRead'] = test_helper_1.createBoolAttr(true);
                            node.params['identicalElementShapes'] = test_helper_1.createBoolAttr(true);
                            node.inputNames = ['input1'];
                            return [4, control_executor_1.executeOp(node, { input1: input1 }, context)];
                        case 1:
                            tensorId = (_a.sent())[0].dataSync()[0];
                            expect(context.getTensorArray(tensorId)).toBeDefined();
                            return [2];
                    }
                });
            }); });
        });
        describe('tensorArrayWrite', function () {
            it('should write the tensor to tensorArray', function () { return __awaiter(_this, void 0, void 0, function () {
                var tensorArray, input2, input3;
                return __generator(this, function (_a) {
                    switch (_a.label) {
                        case 0:
                            tensorArray = new tensor_array_1.TensorArray('', 'int32', 5, [], true, false, true);
                            context.addTensorArray(tensorArray);
                            node.op = 'tensorArrayWrite';
                            node.params['tensorArrayId'] = test_helper_1.createNumberAttrFromIndex(0);
                            node.params['index'] = test_helper_1.createNumberAttrFromIndex(1);
                            node.params['tensor'] = test_helper_1.createTensorAttr(2);
                            node.inputNames = ['input2', 'input3', 'input1'];
                            input2 = [tfjs_core_1.scalar(tensorArray.id)];
                            input3 = [tfjs_core_1.scalar(0)];
                            return [4, control_executor_1.executeOp(node, { input1: input1, input2: input2, input3: input3 }, context)];
                        case 1:
                            _a.sent();
                            expect(tensorArray.size()).toEqual(1);
                            return [2];
                    }
                });
            }); });
        });
        describe('tensorArrayRead', function () {
            it('should read the tensor from tensorArray', function () { return __awaiter(_this, void 0, void 0, function () {
                var tensorArray, input4, input2, input3, read;
                return __generator(this, function (_a) {
                    switch (_a.label) {
                        case 0:
                            tensorArray = new tensor_array_1.TensorArray('', 'int32', 5, [3], true, false, true);
                            input4 = tfjs_core_1.tensor1d([0, 0, 0], 'int32');
                            tensorArray.write(0, input4);
                            context.addTensorArray(tensorArray);
                            node.op = 'tensorArrayRead';
                            node.params['tensorArrayId'] = test_helper_1.createNumberAttrFromIndex(0);
                            node.params['index'] = test_helper_1.createNumberAttrFromIndex(1);
                            node.inputNames = ['input2', 'input3'];
                            input2 = [tfjs_core_1.scalar(tensorArray.id)];
                            input3 = [tfjs_core_1.scalar(0)];
                            return [4, control_executor_1.executeOp(node, { input1: input1, input2: input2, input3: input3 }, context)];
                        case 1:
                            read = _a.sent();
                            test_util_1.expectArraysClose(read[0], input4);
                            return [2];
                    }
                });
            }); });
        });
        describe('tensorArrayGather', function () {
            it('should gather the tensors from tensorArray', function () { return __awaiter(_this, void 0, void 0, function () {
                var tensorArray, input4, input5, input2, input3, gather;
                return __generator(this, function (_a) {
                    switch (_a.label) {
                        case 0:
                            tensorArray = new tensor_array_1.TensorArray('', 'int32', 5, [3], true, false, true);
                            input4 = tfjs_core_1.tensor1d([0, 0, 0], 'int32');
                            input5 = tfjs_core_1.tensor1d([1, 1, 1], 'int32');
                            tensorArray.writeMany([0, 1], [input4, input5]);
                            context.addTensorArray(tensorArray);
                            node.op = 'tensorArrayGather';
                            node.params['tensorArrayId'] = test_helper_1.createNumberAttrFromIndex(0);
                            node.params['indices'] = test_helper_1.createNumericArrayAttrFromIndex(1);
                            node.params['dtype'] = test_helper_1.createDtypeAttr('int32');
                            node.inputNames = ['input2', 'input3'];
                            input2 = [tfjs_core_1.scalar(tensorArray.id)];
                            input3 = [tfjs_core_1.tensor1d([0, 1])];
                            return [4, control_executor_1.executeOp(node, { input2: input2, input3: input3 }, context)];
                        case 1:
                            gather = _a.sent();
                            expect(gather.length).toEqual(1);
                            expect(gather[0].shape).toEqual([2, 3]);
                            test_util_1.expectArraysClose(gather[0].dataSync(), new Int32Array([0, 0, 0, 1, 1, 1]));
                            return [2];
                    }
                });
            }); });
        });
        describe('tensorArrayScatter', function () {
            it('should scatter the tensor to tensorArray', function () { return __awaiter(_this, void 0, void 0, function () {
                var tensorArray, input4, input2, input3;
                return __generator(this, function (_a) {
                    switch (_a.label) {
                        case 0:
                            tensorArray = new tensor_array_1.TensorArray('', 'int32', 5, [3], true, false, true);
                            input4 = [tfjs_core_1.tensor2d([0, 0, 0, 1, 1, 1], [2, 3], 'int32')];
                            context.addTensorArray(tensorArray);
                            node.op = 'tensorArrayScatter';
                            node.params['tensorArrayId'] = test_helper_1.createNumberAttrFromIndex(0);
                            node.params['indices'] = test_helper_1.createNumericArrayAttrFromIndex(1);
                            node.params['tensor'] = test_helper_1.createTensorAttr(2);
                            node.inputNames = ['input2', 'input3', 'input4'];
                            input2 = [tfjs_core_1.scalar(tensorArray.id)];
                            input3 = [tfjs_core_1.tensor1d([0, 1], 'int32')];
                            return [4, control_executor_1.executeOp(node, { input2: input2, input3: input3, input4: input4 }, context)];
                        case 1:
                            _a.sent();
                            expect(tensorArray.size()).toEqual(2);
                            return [2];
                    }
                });
            }); });
        });
        describe('tensorArraySplit', function () {
            it('should split the tensor to tensorArray', function () { return __awaiter(_this, void 0, void 0, function () {
                var tensorArray, input4, input2, input3;
                return __generator(this, function (_a) {
                    switch (_a.label) {
                        case 0:
                            tensorArray = new tensor_array_1.TensorArray('', 'int32', 2, [3], true, false, true);
                            input4 = [tfjs_core_1.tensor2d([0, 0, 0, 1, 1, 1], [2, 3], 'int32')];
                            context.addTensorArray(tensorArray);
                            node.op = 'tensorArraySplit';
                            node.params['tensorArrayId'] = test_helper_1.createNumberAttrFromIndex(0);
                            node.params['tensor'] = test_helper_1.createTensorAttr(1);
                            node.params['lengths'] = test_helper_1.createNumericArrayAttrFromIndex(2);
                            node.inputNames = ['input2', 'input4', 'input3'];
                            input2 = [tfjs_core_1.scalar(tensorArray.id)];
                            input3 = [tfjs_core_1.tensor1d([1, 1], 'int32')];
                            return [4, control_executor_1.executeOp(node, { input2: input2, input3: input3, input4: input4 }, context)];
                        case 1:
                            _a.sent();
                            expect(tensorArray.size()).toEqual(2);
                            return [2];
                    }
                });
            }); });
        });
        describe('tensorArrayConcat', function () {
            it('should concat the tensors from tensorArray', function () { return __awaiter(_this, void 0, void 0, function () {
                var tensorArray, input4, input5, input2, concat;
                return __generator(this, function (_a) {
                    switch (_a.label) {
                        case 0:
                            tensorArray = new tensor_array_1.TensorArray('', 'int32', 5, [3], true, false, true);
                            input4 = tfjs_core_1.tensor1d([0, 0, 0], 'int32');
                            input5 = tfjs_core_1.tensor1d([1, 1, 1], 'int32');
                            tensorArray.writeMany([0, 1], [input4, input5]);
                            context.addTensorArray(tensorArray);
                            node.op = 'tensorArrayConcat';
                            node.params['tensorArrayId'] = test_helper_1.createNumberAttrFromIndex(0);
                            node.params['dtype'] = test_helper_1.createDtypeAttr('int32');
                            node.inputNames = ['input2'];
                            input2 = [tfjs_core_1.scalar(tensorArray.id)];
                            return [4, control_executor_1.executeOp(node, { input2: input2 }, context)];
                        case 1:
                            concat = _a.sent();
                            expect(concat.length).toEqual(1);
                            expect(concat[0].shape).toEqual([6]);
                            test_util_1.expectArraysClose(concat[0].dataSync(), new Int32Array([0, 0, 0, 1, 1, 1]));
                            return [2];
                    }
                });
            }); });
        });
        describe('tensorArraySize', function () {
            it('should get the size of tensorArray', function () { return __awaiter(_this, void 0, void 0, function () {
                var tensorArray, input4, input5, input2, size;
                return __generator(this, function (_a) {
                    switch (_a.label) {
                        case 0:
                            tensorArray = new tensor_array_1.TensorArray('', 'int32', 5, [3], true, false, true);
                            input4 = tfjs_core_1.tensor1d([0, 0, 0], 'int32');
                            input5 = tfjs_core_1.tensor1d([1, 1, 1], 'int32');
                            tensorArray.writeMany([0, 1], [input4, input5]);
                            context.addTensorArray(tensorArray);
                            node.op = 'tensorArraySize';
                            node.params['tensorArrayId'] = test_helper_1.createNumberAttrFromIndex(0);
                            node.inputNames = ['input2'];
                            input2 = [tfjs_core_1.scalar(tensorArray.id)];
                            return [4, control_executor_1.executeOp(node, { input2: input2 }, context)];
                        case 1:
                            size = _a.sent();
                            expect(size.length).toEqual(1);
                            expect(size[0].shape).toEqual([]);
                            test_util_1.expectArraysClose(size[0].dataSync(), new Int32Array([2]));
                            return [2];
                    }
                });
            }); });
        });
        describe('tensorArrayClose', function () {
            it('should close the tensorArray', function () { return __awaiter(_this, void 0, void 0, function () {
                var tensorArray, input4, input5, input2;
                return __generator(this, function (_a) {
                    switch (_a.label) {
                        case 0:
                            tensorArray = new tensor_array_1.TensorArray('', 'int32', 5, [3], true, false, true);
                            input4 = tfjs_core_1.tensor1d([0, 0, 0], 'int32');
                            input5 = tfjs_core_1.tensor1d([1, 1, 1], 'int32');
                            tensorArray.writeMany([0, 1], [input4, input5]);
                            context.addTensorArray(tensorArray);
                            node.op = 'tensorArrayClose';
                            node.params['tensorArrayId'] = test_helper_1.createNumberAttrFromIndex(0);
                            node.inputNames = ['input2'];
                            input2 = [tfjs_core_1.scalar(tensorArray.id)];
                            return [4, control_executor_1.executeOp(node, { input2: input2 }, context)];
                        case 1:
                            _a.sent();
                            expect(tensorArray.closed).toBeTruthy();
                            return [2];
                    }
                });
            }); });
        });
    });
});
//# sourceMappingURL=control_executor_test.js.map