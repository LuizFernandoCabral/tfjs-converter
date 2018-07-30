"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var tfc = require("@tensorflow/tfjs-core");
var execution_context_1 = require("../../executor/execution_context");
var evaluation_executor_1 = require("./evaluation_executor");
var test_helper_1 = require("./test_helper");
describe('evaluation', function () {
    var node;
    var input1 = [tfc.tensor1d([1])];
    var input2 = [tfc.scalar(1)];
    var context = new execution_context_1.ExecutionContext({}, {});
    beforeEach(function () {
        node = {
            name: 'input1',
            op: '',
            category: 'evaluation',
            inputNames: ['input1', 'input2'],
            inputs: [],
            params: {},
            children: []
        };
    });
    describe('executeOp', function () {
        describe('topK', function () {
            it('should return input', function () {
                node.op = 'topK';
                node.params['x'] = test_helper_1.createTensorAttr(0);
                node.params['k'] = test_helper_1.createNumberAttrFromIndex(1);
                node.params['sorted'] = test_helper_1.createBoolAttr(true);
                spyOn(tfc, 'topk').and.callThrough();
                evaluation_executor_1.executeOp(node, { input1: input1, input2: input2 }, context);
                expect(tfc.topk).toHaveBeenCalledWith(input1[0], 1, true);
            });
        });
    });
});
//# sourceMappingURL=evaluation_executor_test.js.map