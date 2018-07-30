"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var tfc = require("@tensorflow/tfjs-core");
var execution_context_1 = require("../../executor/execution_context");
var test_helper_1 = require("./test_helper");
var transformation_executor_1 = require("./transformation_executor");
describe('transformation', function () {
    var node;
    var input1 = [tfc.scalar(1)];
    var input2 = [tfc.tensor1d([1, 1])];
    var context = new execution_context_1.ExecutionContext({}, {});
    beforeEach(function () {
        node = {
            name: 'test',
            op: '',
            category: 'transformation',
            inputNames: ['input1'],
            inputs: [],
            params: { x: test_helper_1.createTensorAttr(0) },
            children: []
        };
    });
    describe('executeOp', function () {
        describe('cast', function () {
            it('should call tfc.cast', function () {
                spyOn(tfc, 'cast');
                node.op = 'cast';
                node.params.dtype = test_helper_1.createDtypeAttr('float32');
                transformation_executor_1.executeOp(node, { input1: input1 }, context);
                expect(tfc.cast).toHaveBeenCalledWith(input1[0], 'float32');
            });
        });
        describe('expandDims', function () {
            it('should call tfc.expandDims', function () {
                spyOn(tfc, 'expandDims');
                node.op = 'expandDims';
                node.params.axis = test_helper_1.createNumberAttr(1);
                transformation_executor_1.executeOp(node, { input1: input1 }, context);
                expect(tfc.expandDims).toHaveBeenCalledWith(input1[0], 1);
            });
        });
        describe('pad', function () {
            it('should call tfc.pad', function () {
                spyOn(tfc, 'pad');
                node.op = 'pad';
                node.params.padding = test_helper_1.createNumericArrayAttrFromIndex(1);
                node.params.constantValue = test_helper_1.createNumberAttr(1);
                node.inputNames = ['input1', 'input3'];
                var input3 = [tfc.tensor2d([1, 1, 2, 2], [2, 2])];
                transformation_executor_1.executeOp(node, { input1: input1, input3: input3 }, context);
                expect(tfc.pad).toHaveBeenCalledWith(input1[0], [[1, 1], [2, 2]], 1);
            });
        });
        describe('reshape', function () {
            it('should call tfc.reshape', function () {
                spyOn(tfc, 'reshape');
                node.op = 'reshape';
                node.params.shape = test_helper_1.createNumericArrayAttrFromIndex(1);
                node.inputNames = ['input1', 'input2'];
                transformation_executor_1.executeOp(node, { input1: input1, input2: input2 }, context);
                expect(tfc.reshape).toHaveBeenCalledWith(input1[0], [1, 1]);
            });
        });
        describe('squeeze', function () {
            it('should call tfc.squeeze', function () {
                spyOn(tfc, 'squeeze');
                node.op = 'squeeze';
                node.params.axis = test_helper_1.createNumberAttr(1);
                transformation_executor_1.executeOp(node, { input1: input1 }, context);
                expect(tfc.squeeze).toHaveBeenCalledWith(input1[0], 1);
            });
        });
        describe('spaceToBatchND', function () {
            it('should call tfc.spaceToBatchND', function () {
                spyOn(tfc, 'spaceToBatchND');
                node.op = 'spaceToBatchND';
                node.params.blockShape = test_helper_1.createNumericArrayAttrFromIndex(1);
                node.params.paddings = test_helper_1.createNumericArrayAttrFromIndex(2);
                node.inputNames = ['input1', 'input2', 'input3'];
                var input2 = [tfc.tensor1d([1, 1, 2, 2])];
                var input3 = [tfc.tensor2d([1, 2, 2, 3, 2, 3, 3, 4], [4, 2])];
                transformation_executor_1.executeOp(node, { input1: input1, input2: input2, input3: input3 }, context);
                expect(tfc.spaceToBatchND)
                    .toHaveBeenCalledWith(input1[0], [1, 1, 2, 2], [[1, 2], [2, 3], [2, 3], [3, 4]]);
            });
        });
        describe('batchToSpaceND', function () {
            it('should call tfc.batchToSpaceND', function () {
                spyOn(tfc, 'batchToSpaceND');
                node.op = 'batchToSpaceND';
                node.params.blockShape = test_helper_1.createNumericArrayAttrFromIndex(1);
                node.params.crops = test_helper_1.createNumericArrayAttrFromIndex(2);
                node.inputNames = ['input1', 'input2', 'input3'];
                var input2 = [tfc.tensor1d([1, 1, 2, 2])];
                var input3 = [tfc.tensor2d([1, 2, 2, 3, 2, 3, 3, 4], [4, 2])];
                transformation_executor_1.executeOp(node, { input1: input1, input2: input2, input3: input3 }, context);
                expect(tfc.batchToSpaceND)
                    .toHaveBeenCalledWith(input1[0], [1, 1, 2, 2], [[1, 2], [2, 3], [2, 3], [3, 4]]);
            });
        });
    });
});
//# sourceMappingURL=transformation_executor_test.js.map