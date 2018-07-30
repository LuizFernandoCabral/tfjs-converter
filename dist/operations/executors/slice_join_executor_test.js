"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var tfc = require("@tensorflow/tfjs-core");
var execution_context_1 = require("../../executor/execution_context");
var slice_join_executor_1 = require("./slice_join_executor");
var test_helper_1 = require("./test_helper");
describe('slice join', function () {
    var node;
    var input1 = [tfc.scalar(1)];
    var input2 = [tfc.scalar(2)];
    var input3 = [tfc.scalar(3)];
    var input4 = [tfc.tensor1d([3])];
    var input5 = [tfc.tensor1d([3, 4])];
    var context = new execution_context_1.ExecutionContext({}, {});
    describe('multi-tensor ops', function () {
        beforeEach(function () {
            node = {
                name: 'test',
                op: '',
                category: 'slice_join',
                inputNames: ['input1', 'input2', 'input3'],
                inputs: [],
                params: {
                    tensors: test_helper_1.createTensorsAttr(0, 1),
                    axis: test_helper_1.createNumberAttrFromIndex(-1)
                },
                children: []
            };
        });
        describe('executeOp', function () {
            it('should call tfc.concat', function () {
                var spy = spyOn(tfc, 'concat');
                node.op = 'concat';
                slice_join_executor_1.executeOp(node, { input1: input1, input2: input2, input3: input3 }, context);
                expect(spy).toHaveBeenCalledWith([input1[0], input2[0]], 3);
            });
            it('should call tfc.unstack', function () {
                var spy = spyOn(tfc, 'unstack');
                node.op = 'unstack';
                node.params.tensor = test_helper_1.createTensorAttr(0);
                node.params.axis = test_helper_1.createNumberAttr(4);
                slice_join_executor_1.executeOp(node, { input1: input1 }, context);
                expect(spy).toHaveBeenCalledWith(input1[0], 4);
            });
            it('should call tfc.stack', function () {
                var spy = spyOn(tfc, 'stack');
                node.op = 'stack';
                node.params.tensors = test_helper_1.createTensorsAttr(0, 0);
                node.params.axis = test_helper_1.createNumberAttr(4);
                slice_join_executor_1.executeOp(node, { input1: input1, input2: input2, input3: input3 }, context);
                expect(spy.calls.mostRecent().args[0][0]).toEqual(input1[0]);
                expect(spy.calls.mostRecent().args[0][1]).toEqual(input2[0]);
                expect(spy.calls.mostRecent().args[0][2]).toEqual(input3[0]);
                expect(spy.calls.mostRecent().args[1]).toEqual(4);
            });
            it('should reshape tensors for tfc.stack', function () {
                var spy = spyOn(tfc, 'stack');
                node.op = 'stack';
                node.inputNames = ['input1', 'input2', 'input3', 'input4'];
                node.params.tensors = test_helper_1.createTensorsAttr(0, 0);
                node.params.axis = test_helper_1.createNumberAttr(4);
                slice_join_executor_1.executeOp(node, { input1: input1, input2: input2, input3: input3, input4: input4 }, context);
                expect(spy.calls.mostRecent().args[0][0]).toEqual(input1[0]);
                expect(spy.calls.mostRecent().args[0][1]).toEqual(input2[0]);
                expect(spy.calls.mostRecent().args[0][2]).toEqual(input3[0]);
                expect(spy.calls.mostRecent().args[0][3].shape).toEqual([]);
                expect(spy.calls.mostRecent().args[1]).toEqual(4);
            });
            it('should raise error if tensors shape does not match for tfc.stack', function () {
                node.op = 'stack';
                node.inputNames = ['input1', 'input2', 'input3', 'input5'];
                node.params.tensors = test_helper_1.createTensorsAttr(0, 0);
                node.params.axis = test_helper_1.createNumberAttr(4);
                expect(function () { return slice_join_executor_1.executeOp(node, { input1: input1, input2: input2, input3: input3, input5: input5 }, context); })
                    .toThrow(new Error('the input tensors shape does not match'));
            });
        });
    });
    describe('single-tensor ops', function () {
        beforeEach(function () {
            node = {
                name: 'test',
                op: '',
                category: 'slice_join',
                inputNames: ['input1'],
                inputs: [],
                params: { x: test_helper_1.createTensorAttr(0) },
                children: []
            };
        });
        describe('executeOp', function () {
            it('should call tfc.reverse', function () {
                spyOn(tfc, 'reverse');
                node.op = 'reverse';
                node.params.axis = test_helper_1.createNumberAttrFromIndex(1);
                node.inputNames = ['input1', 'input2'];
                slice_join_executor_1.executeOp(node, { input1: input1, input2: input2 }, context);
                expect(tfc.reverse).toHaveBeenCalledWith(input1[0], 2);
            });
            it('should call tfc.tile', function () {
                spyOn(tfc, 'tile');
                node.op = 'tile';
                node.params.reps = test_helper_1.createNumberAttrFromIndex(1);
                node.inputNames = ['input1', 'input2'];
                slice_join_executor_1.executeOp(node, { input1: input1, input2: input2 }, context);
                expect(tfc.tile).toHaveBeenCalledWith(input1[0], 2);
            });
            it('should call tfc.slice', function () {
                spyOn(tfc, 'slice');
                node.op = 'slice';
                node.params.begin = test_helper_1.createNumericArrayAttr([1]);
                node.params.size = test_helper_1.createNumericArrayAttr([2]);
                slice_join_executor_1.executeOp(node, { input1: input1 }, context);
                expect(tfc.slice).toHaveBeenCalledWith(input1[0], [1], [2]);
            });
            it('should call tfc.stridedSlice', function () {
                spyOn(tfc, 'stridedSlice');
                node.op = 'stridedSlice';
                node.params.begin = test_helper_1.createNumericArrayAttr([1]);
                node.params.end = test_helper_1.createNumericArrayAttr([2]);
                node.params.strides = test_helper_1.createNumericArrayAttr([3]);
                node.params.beginMask = test_helper_1.createNumberAttr(4);
                node.params.endMask = test_helper_1.createNumberAttr(5);
                slice_join_executor_1.executeOp(node, { input1: input1 }, context);
                expect(tfc.stridedSlice)
                    .toHaveBeenCalledWith(input1[0], [1], [2], [3], 4, 5);
            });
            it('should call tfc.gather', function () {
                spyOn(tfc, 'gather');
                node.op = 'gather';
                node.params.axis = test_helper_1.createNumberAttr(1);
                node.params.indices = test_helper_1.createTensorAttr(1);
                node.inputNames = ['input1', 'input2'];
                slice_join_executor_1.executeOp(node, { input1: input1, input2: input2 }, context);
                expect(tfc.gather).toHaveBeenCalledWith(input1[0], input2[0], 1);
            });
            it('should call tfc.split', function () {
                spyOn(tfc, 'split');
                node.op = 'split';
                node.params.axis = test_helper_1.createNumberAttrFromIndex(0);
                node.params.x = test_helper_1.createTensorAttr(1);
                node.params.numOrSizeSplits = test_helper_1.createNumberAttr(2);
                node.inputNames = ['input1', 'input2'];
                slice_join_executor_1.executeOp(node, { input1: input1, input2: input2 }, context);
                expect(tfc.split).toHaveBeenCalledWith(input2[0], 2, 1);
            });
        });
    });
});
//# sourceMappingURL=slice_join_executor_test.js.map