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
var compiled_api_1 = require("../data/compiled_api");
var frozen_model_1 = require("./frozen_model");
var MODEL_URL = 'http://example.org/model.pb';
var WEIGHT_MANIFEST_URL = 'http://example.org/weights_manifest.json';
var RELATIVE_MODEL_URL = '/path/model.pb';
var RELATIVE_WEIGHT_MANIFEST_URL = '/path/weights_manifest.json';
var model;
var bias = tfc.tensor1d([1], 'int32');
var WEIGHT_MAP = {
    'Const': bias
};
var SIMPLE_MODEL = {
    node: [
        {
            name: 'Input',
            op: 'Placeholder',
            attr: {
                dtype: {
                    type: compiled_api_1.tensorflow.DataType.DT_INT32,
                },
                shape: { shape: { dim: [{ size: -1 }, { size: 1 }] } }
            }
        },
        {
            name: 'Const',
            op: 'Const',
            attr: {
                dtype: { type: compiled_api_1.tensorflow.DataType.DT_INT32 },
                value: {
                    tensor: {
                        dtype: compiled_api_1.tensorflow.DataType.DT_INT32,
                        tensorShape: { dim: [{ size: 1 }] },
                    }
                },
                index: { i: 0 },
                length: { i: 4 }
            }
        },
        { name: 'Add', op: 'Add', input: ['Input', 'Const'], attr: {} }
    ],
    versions: { producer: 1.0, minConsumer: 3 }
};
describe('Model', function () {
    beforeEach(function () {
        spyOn(compiled_api_1.tensorflow.GraphDef, 'decode').and.returnValue(SIMPLE_MODEL);
        var weightPromise = new Promise((function (resolve) { return resolve(WEIGHT_MAP); }));
        spyOn(tfc.io, 'loadWeights').and.returnValue(weightPromise);
        model = new frozen_model_1.FrozenModel(MODEL_URL, WEIGHT_MANIFEST_URL);
        spyOn(window, 'fetch')
            .and.callFake(function () { return new Promise((function (resolve) {
            return resolve(new Response(JSON.stringify({ json: 'ok!' })));
        })); });
    });
    afterEach(function () { });
    it('load', function () { return __awaiter(_this, void 0, void 0, function () {
        var loaded;
        return __generator(this, function (_a) {
            switch (_a.label) {
                case 0: return [4, model.load()];
                case 1:
                    loaded = _a.sent();
                    expect(loaded).toBe(true);
                    return [2];
            }
        });
    }); });
    describe('getPathPrefix', function () {
        it('should set pathPrefix (absolute path)', function () { return __awaiter(_this, void 0, void 0, function () {
            return __generator(this, function (_a) {
                model = new frozen_model_1.FrozenModel(MODEL_URL, WEIGHT_MANIFEST_URL);
                expect(model.getPathPrefix()).toEqual('http://example.org/');
                return [2];
            });
        }); });
        it('should set pathPrefix (relative path)', function () { return __awaiter(_this, void 0, void 0, function () {
            return __generator(this, function (_a) {
                model = new frozen_model_1.FrozenModel(RELATIVE_MODEL_URL, RELATIVE_WEIGHT_MANIFEST_URL);
                expect(model.getPathPrefix()).toEqual('/path/');
                return [2];
            });
        }); });
    });
    describe('predict', function () {
        it('should generate the output for single tensor', function () { return __awaiter(_this, void 0, void 0, function () {
            var input, output;
            return __generator(this, function (_a) {
                switch (_a.label) {
                    case 0: return [4, model.load()];
                    case 1:
                        _a.sent();
                        input = tfc.tensor2d([1, 1], [2, 1], 'int32');
                        output = model.predict(input);
                        expect(output.dataSync()[0]).toEqual(2);
                        return [2];
                }
            });
        }); });
        it('should generate the output for tensor array', function () { return __awaiter(_this, void 0, void 0, function () {
            var input, output;
            return __generator(this, function (_a) {
                switch (_a.label) {
                    case 0: return [4, model.load()];
                    case 1:
                        _a.sent();
                        input = tfc.tensor2d([1, 1], [2, 1], 'int32');
                        output = model.predict([input]);
                        expect(output.dataSync()[0]).toEqual(2);
                        return [2];
                }
            });
        }); });
        it('should generate the output for tensor map', function () { return __awaiter(_this, void 0, void 0, function () {
            var input, output;
            return __generator(this, function (_a) {
                switch (_a.label) {
                    case 0: return [4, model.load()];
                    case 1:
                        _a.sent();
                        input = tfc.tensor2d([1, 1], [2, 1], 'int32');
                        output = model.predict({ 'Input': input });
                        expect(output.dataSync()[0]).toEqual(2);
                        return [2];
                }
            });
        }); });
        it('should throw error if input size mismatch', function () { return __awaiter(_this, void 0, void 0, function () {
            var input;
            return __generator(this, function (_a) {
                switch (_a.label) {
                    case 0: return [4, model.load()];
                    case 1:
                        _a.sent();
                        input = tfc.tensor1d([1], 'int32');
                        expect(function () { return model.predict([input, input]); }).toThrow();
                        return [2];
                }
            });
        }); });
        it('should throw exception if inputs shapes do not match', function () {
            var input = tfc.tensor2d([1, 1], [1, 2], 'int32');
            expect(function () { return model.predict([input]); }).toThrow();
        });
        it('should throw exception if inputs dtype does not match graph', function () {
            var input = tfc.tensor1d([1], 'float32');
            expect(function () { return model.predict([input]); }).toThrow();
        });
    });
    describe('execute', function () {
        it('should generate the default output', function () { return __awaiter(_this, void 0, void 0, function () {
            var input, output;
            return __generator(this, function (_a) {
                switch (_a.label) {
                    case 0: return [4, model.load()];
                    case 1:
                        _a.sent();
                        input = tfc.tensor2d([1, 1], [2, 1], 'int32');
                        output = model.execute({ 'Input': input });
                        expect(output.dataSync()[0]).toEqual(2);
                        return [2];
                }
            });
        }); });
        it('should generate the output array', function () { return __awaiter(_this, void 0, void 0, function () {
            var input, output;
            return __generator(this, function (_a) {
                switch (_a.label) {
                    case 0: return [4, model.load()];
                    case 1:
                        _a.sent();
                        input = tfc.tensor2d([1, 1], [2, 1], 'int32');
                        output = model.execute({ 'Input': input }, ['Add', 'Const']);
                        expect(Array.isArray(output)).toBeTruthy();
                        expect(output[0].dataSync()[0]).toEqual(2);
                        expect(output[1].dataSync()[0]).toEqual(1);
                        return [2];
                }
            });
        }); });
        it('should throw exception if inputs shapes do not match', function () {
            var input = tfc.tensor2d([1, 1], [1, 2], 'int32');
            expect(function () { return model.execute([input]); }).toThrow();
        });
        it('should throw exception if inputs dtype does not match graph', function () {
            var input = tfc.tensor2d([1, 1], [2, 1], 'float32');
            expect(function () { return model.predict([input]); }).toThrow();
        });
        it('should throw error if input size mismatch', function () { return __awaiter(_this, void 0, void 0, function () {
            var input;
            return __generator(this, function (_a) {
                switch (_a.label) {
                    case 0: return [4, model.load()];
                    case 1:
                        _a.sent();
                        input = tfc.tensor1d([1], 'int32');
                        expect(function () { return model.execute([input, input]); }).toThrow();
                        return [2];
                }
            });
        }); });
    });
    describe('dispose', function () {
        it('should dispose the weights', function () { return __awaiter(_this, void 0, void 0, function () {
            return __generator(this, function (_a) {
                switch (_a.label) {
                    case 0:
                        model = new frozen_model_1.FrozenModel(MODEL_URL, WEIGHT_MANIFEST_URL);
                        spyOn(bias, 'dispose');
                        return [4, model.load()];
                    case 1:
                        _a.sent();
                        model.dispose();
                        expect(bias.dispose).toHaveBeenCalled();
                        return [2];
                }
            });
        }); });
    });
    describe('getVersion', function () {
        it('should return the version info from the tf model', function () { return __awaiter(_this, void 0, void 0, function () {
            return __generator(this, function (_a) {
                switch (_a.label) {
                    case 0: return [4, model.load()];
                    case 1:
                        _a.sent();
                        expect(model.modelVersion).toEqual('1.3');
                        return [2];
                }
            });
        }); });
    });
    describe('relative path', function () {
        beforeEach(function () {
            model = new frozen_model_1.FrozenModel(RELATIVE_MODEL_URL, RELATIVE_WEIGHT_MANIFEST_URL);
        });
        it('load', function () { return __awaiter(_this, void 0, void 0, function () {
            var loaded;
            return __generator(this, function (_a) {
                switch (_a.label) {
                    case 0: return [4, model.load()];
                    case 1:
                        loaded = _a.sent();
                        expect(loaded).toBe(true);
                        return [2];
                }
            });
        }); });
    });
    it('should loadFrozenModel', function () { return __awaiter(_this, void 0, void 0, function () {
        var model;
        return __generator(this, function (_a) {
            switch (_a.label) {
                case 0: return [4, frozen_model_1.loadFrozenModel(MODEL_URL, WEIGHT_MANIFEST_URL)];
                case 1:
                    model = _a.sent();
                    expect(model).not.toBeUndefined();
                    return [2];
            }
        });
    }); });
    it('should loadFrozenModel with request options', function () { return __awaiter(_this, void 0, void 0, function () {
        var model;
        return __generator(this, function (_a) {
            switch (_a.label) {
                case 0: return [4, frozen_model_1.loadFrozenModel(MODEL_URL, WEIGHT_MANIFEST_URL, { credentials: 'include' })];
                case 1:
                    model = _a.sent();
                    expect(window.fetch).toHaveBeenCalledWith(MODEL_URL, {
                        credentials: 'include'
                    });
                    expect(window.fetch).toHaveBeenCalledWith(WEIGHT_MANIFEST_URL, {
                        credentials: 'include'
                    });
                    expect(model).not.toBeUndefined();
                    return [2];
            }
        });
    }); });
    describe('InferenceModel interface', function () {
        it('should expose inputs', function () { return __awaiter(_this, void 0, void 0, function () {
            return __generator(this, function (_a) {
                switch (_a.label) {
                    case 0: return [4, model.load()];
                    case 1:
                        _a.sent();
                        expect(model.inputs).toEqual([
                            { name: 'Input', shape: [-1, 1], dtype: 'int32' }
                        ]);
                        return [2];
                }
            });
        }); });
        it('should expose outputs', function () { return __awaiter(_this, void 0, void 0, function () {
            return __generator(this, function (_a) {
                switch (_a.label) {
                    case 0: return [4, model.load()];
                    case 1:
                        _a.sent();
                        expect(model.outputs).toEqual([
                            { name: 'Add', shape: undefined, dtype: undefined }
                        ]);
                        return [2];
                }
            });
        }); });
    });
});
//# sourceMappingURL=frozen_model_test.js.map