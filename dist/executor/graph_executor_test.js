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
var test_helper_1 = require("../operations/executors/test_helper");
var operations = require("../operations/operation_executor");
var execution_context_1 = require("./execution_context");
var graph_executor_1 = require("./graph_executor");
var executor;
var inputNode;
var constNode;
var outputNode;
var graph;
var graphWithControlFlow;
var constTensor;
describe('GraphExecutor', function () {
    beforeEach(function () {
        inputNode = {
            inputNames: [],
            inputs: [],
            children: [],
            name: 'input',
            op: 'placeholder',
            category: 'graph',
            params: {}
        };
        constNode = {
            inputNames: [],
            inputs: [],
            children: [],
            name: 'const',
            op: 'const',
            category: 'graph',
            params: {}
        };
        outputNode = {
            inputNames: ['input', 'const'],
            inputs: [inputNode, constNode],
            children: [],
            name: 'output',
            op: 'add',
            category: 'arithmetic',
            params: { 'a': test_helper_1.createTensorAttr(0), 'b': test_helper_1.createTensorAttr(1) }
        };
        graph = {
            inputs: [constNode, inputNode],
            nodes: { 'input': inputNode, 'const': constNode, 'output': outputNode },
            outputs: [outputNode],
            withControlFlow: false,
            withDynamicShape: false,
            placeholders: [inputNode]
        };
        inputNode.children.push(outputNode);
        constNode.children.push(outputNode);
        executor = new graph_executor_1.GraphExecutor(graph);
        constTensor = tfc.scalar(2);
        executor.weightMap = { 'const': [constTensor] };
    });
    afterEach(function () { });
    describe('execute graph', function () {
        describe('initialization', function () {
            it('should expose placehoder names', function () {
                expect(executor.inputNodes).toEqual(['input']);
            });
            it('should expose output names', function () {
                expect(executor.outputNodes).toEqual(['output']);
            });
            it('should expose placeholders', function () {
                inputNode.params['shape'] = { value: [1], type: 'shape' };
                inputNode.params['dtype'] = { value: 'float32', type: 'dtype' };
                expect(executor.inputs).toEqual([
                    { name: 'input', shape: [1], dtype: 'float32' }
                ]);
            });
            it('should expose outputs', function () {
                outputNode.params['shape'] = { value: [1, 1], type: 'shape' };
                outputNode.params['dtype'] = { value: 'int32', type: 'dtype' };
                expect(executor.outputs).toEqual([
                    { name: 'output', shape: [1, 1], dtype: 'int32' }
                ]);
            });
        });
        describe('graph level', function () {
            describe('execute', function () {
                it('should throw exception if missing inputs', function () {
                    expect(function () { return executor.execute({}); })
                        .toThrow(new Error('The dict provided in model.execute(dict) has the keys [], ' +
                        'but is missing the required keys: [input].'));
                });
                it('should throw exception if contains extra inputs', function () {
                    var inputTensor = tfc.scalar(1);
                    expect(function () {
                        return executor.execute({ test: [inputTensor], input: [inputTensor] });
                    })
                        .toThrow(new Error('The dict provided in model.execute(dict) has unused keys: ' +
                        '[test]. Please provide only the following keys: [input].'));
                });
                it('should execute the op', function () {
                    executor = new graph_executor_1.GraphExecutor(graph);
                    var inputTensor = tfc.scalar(1);
                    var spy = spyOn(operations, 'executeOp').and.callFake(function (node) {
                        return node.op === 'const' ? [constTensor] : [inputTensor];
                    });
                    executor.execute({ input: [inputTensor] });
                    expect(spy.calls.allArgs()).toEqual([
                        [inputNode, jasmine.any(Object), jasmine.any(execution_context_1.ExecutionContext)],
                        [constNode, jasmine.any(Object), jasmine.any(execution_context_1.ExecutionContext)],
                        [outputNode, jasmine.any(Object), jasmine.any(execution_context_1.ExecutionContext)]
                    ]);
                });
                it('should throw exception if inputs shapes do not match graph', function () {
                    inputNode.params['shape'] = { value: [1, 1], type: 'shape' };
                    var inputTensor = tfc.tensor1d([1], 'float32');
                    expect(function () { return executor.execute({ input: [inputTensor] }); })
                        .toThrow(new Error('The shape of dict[\'input\'] provided' +
                        ' in model.execute(dict) must be [1,1], but was [1]'));
                });
                it('should throw exception if inputs dtype do not match graph', function () {
                    inputNode.params['dtype'] = { value: 'int32', type: 'dtype' };
                    var inputTensor = tfc.tensor1d([1], 'float32');
                    expect(function () { return executor.execute({ input: [inputTensor] }); })
                        .toThrow(new Error('The dtype of dict[\'input\'] provided' +
                        ' in model.execute(dict) must be int32, but was float32'));
                });
                it('should not throw exception if inputs shapes is dynamic', function () {
                    inputNode.params['shape'] = { value: [-1, 1, 1, 1], type: 'shape' };
                    var inputTensor = tfc.tensor4d([1, 1], [2, 1, 1, 1], 'float32');
                    expect(function () { return executor.execute({ input: [inputTensor] }); }).not.toThrow();
                });
            });
            describe('executeAsync', function () {
                it('should execute control flow graph', function (done) { return __awaiter(_this, void 0, void 0, function () {
                    var inputTensor, constTensor, spy;
                    return __generator(this, function (_a) {
                        switch (_a.label) {
                            case 0:
                                inputNode = {
                                    inputNames: [],
                                    inputs: [],
                                    children: [],
                                    name: 'input',
                                    op: 'placeholder',
                                    category: 'graph',
                                    params: {}
                                };
                                constNode = {
                                    inputNames: [],
                                    inputs: [],
                                    children: [],
                                    name: 'const',
                                    op: 'const',
                                    category: 'graph',
                                    params: {}
                                };
                                outputNode = {
                                    inputNames: ['input', 'const'],
                                    inputs: [inputNode, constNode],
                                    children: [],
                                    name: 'output',
                                    op: 'switch',
                                    category: 'control',
                                    params: {}
                                };
                                inputNode.children.push(outputNode);
                                constNode.children.push(outputNode);
                                graphWithControlFlow = {
                                    inputs: [constNode, inputNode],
                                    nodes: { 'input': inputNode, 'const': constNode, 'output': outputNode },
                                    outputs: [outputNode],
                                    withControlFlow: true,
                                    withDynamicShape: false,
                                    placeholders: [inputNode]
                                };
                                executor = new graph_executor_1.GraphExecutor(graphWithControlFlow);
                                inputTensor = tfc.scalar(1);
                                constTensor = tfc.scalar(2);
                                executor.weightMap = { const: [constTensor] };
                                spy = spyOn(operations, 'executeOp').and.callFake(function (node) {
                                    return node.op === 'const' ? [constTensor] : [inputTensor];
                                });
                                return [4, executor.executeAsync({ input: [inputTensor] }).then(function (result) {
                                        expect(spy.calls.allArgs()).toEqual([
                                            [inputNode, jasmine.any(Object), jasmine.any(execution_context_1.ExecutionContext)],
                                            [outputNode, jasmine.any(Object), jasmine.any(execution_context_1.ExecutionContext)],
                                            [constNode, jasmine.any(Object), jasmine.any(execution_context_1.ExecutionContext)],
                                        ]);
                                        done();
                                    })];
                            case 1:
                                _a.sent();
                                return [2];
                        }
                    });
                }); });
            });
        });
    });
});
//# sourceMappingURL=graph_executor_test.js.map