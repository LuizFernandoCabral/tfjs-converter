"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var execution_context_1 = require("./execution_context");
var tensor_array_1 = require("./tensor_array");
var context;
var tensorArray;
describe('ExecutionContext', function () {
    beforeEach(function () {
        context = new execution_context_1.ExecutionContext({}, {});
    });
    it('should initialize', function () {
        expect(context.currentContext).toEqual([
            { id: 0, frameName: '', iterationId: 0 }
        ]);
        expect(context.currentContextId).toEqual('');
    });
    describe('tensor array', function () {
        beforeEach(function () {
            tensorArray = new tensor_array_1.TensorArray('', 'float32', 10, [1], true, true, true);
        });
        it('should be able to add tensor array', function () {
            context.addTensorArray(tensorArray);
            expect(context.getTensorArray(tensorArray.id)).toBe(tensorArray);
        });
        it('should be able to read tensor array', function () {
            expect(context.getTensorArray(tensorArray.id)).toBeUndefined();
        });
    });
    describe('enterFrame', function () {
        it('should add new Frame', function () {
            context.enterFrame('1');
            expect(context.currentContextId).toEqual('/1-0');
            expect(context.currentContext).toEqual([
                { id: 0, frameName: '', iterationId: 0 },
                { id: 1, frameName: '1', iterationId: 0 }
            ]);
        });
    });
    describe('exitFrame', function () {
        it('should remove Frame', function () {
            context.enterFrame('1');
            context.exitFrame();
            expect(context.currentContextId).toEqual('');
            expect(context.currentContext).toEqual([
                { id: 0, frameName: '', iterationId: 0 }
            ]);
        });
        it('should remember previous Frame', function () {
            context.enterFrame('1');
            context.nextIteration();
            context.enterFrame('2');
            context.exitFrame();
            expect(context.currentContextId).toEqual('/1-1');
            expect(context.currentContext).toEqual([
                { id: 0, frameName: '', iterationId: 0 },
                { id: 2, frameName: '1', iterationId: 1 }
            ]);
        });
    });
    describe('nextIteration', function () {
        it('should increate iteration', function () {
            context.enterFrame('1');
            context.nextIteration();
            expect(context.currentContextId).toEqual('/1-1');
            expect(context.currentContext).toEqual([
                { id: 0, frameName: '', iterationId: 0 },
                { id: 2, frameName: '1', iterationId: 1 }
            ]);
        });
    });
});
//# sourceMappingURL=execution_context_test.js.map