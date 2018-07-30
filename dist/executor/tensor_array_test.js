"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var tfjs_core_1 = require("@tensorflow/tfjs-core");
var tfjs_core_2 = require("@tensorflow/tfjs-core");
var tensor_array_1 = require("./tensor_array");
var tensorArray;
var tensor;
var tensor2;
var NAME = 'TA1';
var DTYPE = 'int32';
var SIZE = 10;
var SHAPE = [1, 1];
var IDENTICAL_SHAPE = true;
var DYNAMIC_SIZE = false;
var CLEAR_AFTER_READ = false;
describe('TensorArray', function () {
    beforeEach(function () {
        tensorArray = new tensor_array_1.TensorArray(NAME, DTYPE, SIZE, SHAPE, IDENTICAL_SHAPE, DYNAMIC_SIZE, CLEAR_AFTER_READ);
        tensor = tfjs_core_1.tensor2d([1], [1, 1], 'int32');
        tensor2 = tfjs_core_1.tensor2d([2], [1, 1], 'int32');
    });
    afterEach(function () { return tensorArray.clearAndClose(); });
    it('should initialize', function () {
        expect(tensorArray.size()).toEqual(0);
        expect(tensorArray.name).toEqual('TA1');
        expect(tensorArray.dynamicSize).toBeFalsy();
        expect(tensorArray.closed).toBeFalsy();
    });
    it('should close', function () {
        var numOfTensors = tfjs_core_1.memory().numTensors;
        var size = tensorArray.size();
        tensorArray.clearAndClose();
        expect(tensorArray.size()).toBe(0);
        expect(tensorArray.closed).toBeTruthy();
        expect(tfjs_core_1.memory().numTensors).toEqual(numOfTensors - size);
    });
    describe('write', function () {
        it('should add new tensor', function () {
            tensorArray.write(0, tensor);
            expect(tensorArray.size()).toBe(1);
        });
        it('should add multiple tensors', function () {
            tensorArray.writeMany([0, 1], [tensor, tensor2]);
            expect(tensorArray.size()).toBe(2);
        });
        it('should fail if dtype does not match', function () {
            var tensor = tfjs_core_1.tensor2d([1], [1, 1], 'float32');
            expect(function () { return tensorArray.write(0, tensor); }).toThrow();
        });
        it('should fail if shape does not match', function () {
            var tensor = tfjs_core_1.tensor2d([1, 2], [2, 1], 'int32');
            expect(function () { return tensorArray.write(0, tensor); }).toThrow();
        });
        it('should fail if the index has already been written', function () {
            tensorArray.write(0, tensor);
            expect(function () { return tensorArray.write(0, tensor); }).toThrow();
        });
        it('should fail if the index greater than array size', function () {
            expect(function () { return tensorArray.write(11, tensor); }).toThrow();
        });
        it('should fail if the array is closed', function () {
            tensorArray.clearAndClose();
            expect(function () { return tensorArray.write(0, tensor); }).toThrow();
        });
        it('should create no new tensors', function () {
            var numTensors = tfjs_core_1.memory().numTensors;
            tensorArray.write(0, tensor);
            expect(tfjs_core_1.memory().numTensors).toEqual(numTensors);
        });
    });
    describe('read', function () {
        beforeEach(function () {
            tensorArray.writeMany([0, 1], [tensor, tensor2]);
        });
        it('should read the correct index', function () {
            expect(tensorArray.read(0)).toBe(tensor);
            expect(tensorArray.read(1)).toBe(tensor2);
        });
        it('should read the multiple indices', function () {
            expect(tensorArray.readMany([0, 1])).toEqual([tensor, tensor2]);
        });
        it('should failed if index is out of bound', function () {
            expect(function () { return tensorArray.read(3); }).toThrow();
            expect(function () { return tensorArray.read(-1); }).toThrow();
        });
        it('should failed if array is closed', function () {
            tensorArray.clearAndClose();
            expect(function () { return tensorArray.read(0); }).toThrow();
        });
        it('should create no new tensors', function () {
            var numTensors = tfjs_core_1.memory().numTensors;
            tensorArray.read(0);
            tensorArray.read(1);
            expect(tfjs_core_1.memory().numTensors).toEqual(numTensors);
        });
    });
    describe('gather', function () {
        beforeEach(function () {
            tensorArray.writeMany([0, 1], [tensor, tensor2]);
        });
        it('should return default packed tensors', function () {
            var gathered = tensorArray.gather();
            expect(gathered.shape).toEqual([2, 1, 1]);
            tfjs_core_2.test_util.expectArraysClose(gathered, [1, 2]);
        });
        it('should return packed tensors when indices is specified', function () {
            var gathered = tensorArray.gather([1, 0]);
            expect(gathered.shape).toEqual([2, 1, 1]);
            tfjs_core_2.test_util.expectArraysClose(gathered, [2, 1]);
        });
        it('should fail if dtype is not matched', function () {
            expect(function () { return tensorArray.gather([0, 1], 'float32'); }).toThrow();
        });
        it('should create one new tensor', function () {
            var numTensors = tfjs_core_1.memory().numTensors;
            tensorArray.gather();
            expect(tfjs_core_1.memory().numTensors).toEqual(numTensors + 1);
        });
    });
    describe('concat', function () {
        beforeEach(function () {
            tensorArray.writeMany([0, 1], [tensor, tensor2]);
        });
        it('should return default concat tensors', function () {
            var concat = tensorArray.concat();
            expect(concat.shape).toEqual([2, 1]);
            tfjs_core_2.test_util.expectArraysClose(concat, [1, 2]);
        });
        it('should fail if dtype is not matched', function () {
            expect(function () { return tensorArray.concat('float32'); }).toThrow();
        });
        it('should create one new tensor', function () {
            var numTensors = tfjs_core_1.memory().numTensors;
            tensorArray.concat();
            expect(tfjs_core_1.memory().numTensors).toEqual(numTensors + 1);
        });
    });
    describe('scatter', function () {
        it('should scatter the input tensor', function () {
            var input = tfjs_core_1.tensor3d([1, 2, 3], [3, 1, 1], 'int32');
            tensorArray.scatter([0, 1, 2], input);
            expect(tensorArray.size()).toEqual(3);
        });
        it('should fail if indices and tensor shapes do not matched', function () {
            var input = tfjs_core_1.tensor3d([1, 2, 3], [3, 1, 1], 'int32');
            expect(function () { return tensorArray.scatter([1, 2], input); }).toThrow();
        });
        it('should fail if max index > array max size', function () {
            var input = tfjs_core_1.tensor3d([1, 2, 3], [3, 1, 1], 'int32');
            expect(function () { return tensorArray.scatter([1, 2, 11], input); }).toThrow();
        });
        it('should fail if dtype is not matched', function () {
            var input = tfjs_core_1.tensor3d([1, 2, 3], [3, 1, 1], 'float32');
            expect(function () { return tensorArray.scatter([0, 1, 2], input); }).toThrow();
        });
        it('should create three new tensors', function () {
            var input = tfjs_core_1.tensor3d([1, 2, 3], [3, 1, 1], 'int32');
            var numTensors = tfjs_core_1.memory().numTensors;
            tensorArray.scatter([0, 1, 2], input);
            expect(tfjs_core_1.memory().numTensors).toEqual(numTensors + 3);
        });
    });
    describe('split', function () {
        beforeEach(function () {
            tensorArray = new tensor_array_1.TensorArray(NAME, DTYPE, 3, [2, 2], IDENTICAL_SHAPE, DYNAMIC_SIZE, CLEAR_AFTER_READ);
        });
        it('should split the input tensor', function () {
            var input = tfjs_core_1.tensor3d([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], [3, 2, 2], 'int32');
            tensorArray.split([1, 1, 1], input);
            expect(tensorArray.size()).toEqual(3);
        });
        it('should fail if indices and tensor shapes do not matched', function () {
            var input = tfjs_core_1.tensor3d([1, 2, 3], [3, 1, 1], 'int32');
            expect(function () { return tensorArray.split([1, 1], input); }).toThrow();
        });
        it('should fail if length > array max size', function () {
            var input = tfjs_core_1.tensor3d([1, 2, 3], [3, 1, 1], 'int32');
            expect(function () { return tensorArray.split([1, 1, 1, 1], input); }).toThrow();
        });
        it('should fail if dtype is not matched', function () {
            var input = tfjs_core_1.tensor3d([1, 2, 3], [3, 1, 1], 'float32');
            expect(function () { return tensorArray.split([1, 1, 1], input); }).toThrow();
        });
        it('should create three new tensors', function () {
            var input = tfjs_core_1.tensor3d([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], [3, 2, 2], 'int32');
            var numTensors = tfjs_core_1.memory().numTensors;
            tensorArray.split([1, 1, 1], input);
            expect(tfjs_core_1.memory().numTensors).toEqual(numTensors + 3);
        });
    });
});
//# sourceMappingURL=tensor_array_test.js.map