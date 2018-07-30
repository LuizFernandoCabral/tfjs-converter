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
var execution_context_1 = require("../../executor/execution_context");
var dynamic_executor_1 = require("./dynamic_executor");
var test_helper_1 = require("./test_helper");
describe('dynamic', function () {
    var node;
    var input1 = [tfc.tensor1d([1])];
    var context = new execution_context_1.ExecutionContext({}, {});
    beforeEach(function () {
        node = {
            name: 'input1',
            op: '',
            category: 'image',
            inputNames: ['input1'],
            inputs: [],
            params: {},
            children: []
        };
    });
    describe('executeOp', function () {
        describe('nonMaxSuppression', function () {
            it('should return input', function () {
                node.op = 'nonMaxSuppression';
                node.params['boxes'] = test_helper_1.createTensorAttr(0);
                node.params['scores'] = test_helper_1.createTensorAttr(1);
                node.params['maxOutputSize'] = test_helper_1.createTensorAttr(2);
                node.params['iouThreshold'] = test_helper_1.createTensorAttr(3);
                node.params['scoreThreshold'] = test_helper_1.createNumberAttr(1);
                node.inputNames = ['input1', 'input2', 'input3', 'input4'];
                var input2 = [tfc.tensor1d([1])];
                var input3 = [tfc.tensor1d([1])];
                var input4 = [tfc.tensor1d([1])];
                spyOn(tfc.image, 'nonMaxSuppressionAsync').and.callThrough();
                var result = dynamic_executor_1.executeOp(node, { input1: input1, input2: input2, input3: input3, input4: input4 }, context);
                expect(tfc.image.nonMaxSuppressionAsync)
                    .toHaveBeenCalledWith(input1[0], input2[0], input3[0], input4[0], 1);
                expect(result instanceof Promise).toBeTruthy();
            });
        });
        describe('whereAsync', function () {
            it('should call tfc.whereAsync', function () { return __awaiter(_this, void 0, void 0, function () {
                var input1, result;
                return __generator(this, function (_a) {
                    node.op = 'whereAsync';
                    input1 = [tfc.scalar(1)];
                    node.params = { 'condition': test_helper_1.createTensorAttr(0) };
                    spyOn(tfc, 'whereAsync').and.callThrough();
                    result = dynamic_executor_1.executeOp(node, { input1: input1 }, context);
                    expect(tfc.whereAsync).toHaveBeenCalledWith(input1[0]);
                    expect(result instanceof Promise).toBeTruthy();
                    return [2];
                });
            }); });
        });
    });
});
//# sourceMappingURL=dynamic_executor_test.js.map