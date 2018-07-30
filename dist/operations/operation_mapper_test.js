"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var Long = require("long");
var compiled_api_1 = require("../data/compiled_api");
var operation_mapper_1 = require("./operation_mapper");
var mapper = operation_mapper_1.OperationMapper.Instance;
var graph;
var SIMPLE_MODEL = {
    node: [
        {
            name: 'image_placeholder',
            op: 'Placeholder',
            attr: {
                dtype: {
                    type: compiled_api_1.tensorflow.DataType.DT_FLOAT,
                },
                shape: {
                    shape: {
                        dim: [
                            { size: Long.fromInt(3) }, { size: Long.fromInt(3) },
                            { size: Long.fromInt(3) }, { size: Long.fromInt(1) }
                        ]
                    }
                }
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
                        tensorShape: { dim: [{ size: 3 }, { size: 3 }, { size: 1 }, { size: 1 }] },
                        intVal: [0, 0, 0, 0, 1, 0, 0, 0, 0]
                    }
                }
            }
        },
        {
            name: 'Shape',
            op: 'Const',
            attr: {
                dtype: { type: compiled_api_1.tensorflow.DataType.DT_INT32 },
                value: {
                    tensor: {
                        dtype: compiled_api_1.tensorflow.DataType.DT_INT32,
                        tensorShape: { dim: [{ size: 3 }, { size: 1 }, { size: 1 }, { size: 1 }] },
                        intVal: [1, 1, 1]
                    }
                }
            }
        },
        {
            name: 'Value',
            op: 'Const',
            attr: { dtype: { type: compiled_api_1.tensorflow.DataType.DT_INT32 }, value: { i: 1 } }
        },
        { name: 'Fill', op: 'Fill', input: ['Shape', 'Value'], attr: {} }, {
            name: 'Conv2D',
            op: 'Conv2D',
            input: ['image_placeholder', 'Const'],
            attr: {
                T: { type: compiled_api_1.tensorflow.DataType.DT_FLOAT },
                dataFormat: { s: Uint8Array.from([1, 12, 2]) },
                padding: { s: Uint8Array.from([118, 97, 108, 105, 100]) },
                strides: { list: { f: [], i: [1, 2, 2, 1] } },
                useCudnnOnGpu: { b: true }
            }
        },
        {
            name: 'BiasAdd',
            op: 'BiasAdd',
            input: ['Conv2D', 'Shape'],
            attr: {
                T: { type: compiled_api_1.tensorflow.DataType.DT_FLOAT },
                dataFormat: { s: Uint8Array.from([1, 2, 34]) }
            }
        },
        {
            name: 'Squeeze',
            op: 'Squeeze',
            input: ['BiasAdd'],
            attr: { squeeze_dims: { list: { i: [Long.fromInt(1), Long.fromInt(2)] } } }
        }
    ],
    versions: { producer: 1.0 }
};
describe('operationMapper', function () {
    beforeEach(function () {
        graph = mapper.transformGraph(SIMPLE_MODEL);
    });
    afterEach(function () { });
    describe('transform graph', function () {
        describe('graph level', function () {
            it('should find the graph input nodes', function () {
                expect(graph.inputs.map(function (node) { return node.name; })).toEqual([
                    'image_placeholder', 'Const', 'Shape', 'Value'
                ]);
            });
            it('should find the graph output nodes', function () {
                expect(graph.outputs.map(function (node) { return node.name; })).toEqual([
                    'Fill', 'Squeeze'
                ]);
            });
            it('should convert nodes', function () {
                expect(Object.keys(graph.nodes)).toEqual([
                    'image_placeholder', 'Const', 'Shape', 'Value', 'Fill', 'Conv2D',
                    'BiasAdd', 'Squeeze'
                ]);
            });
        });
        describe('node level', function () {
            it('should find the input nodes', function () {
                expect(graph.nodes['Fill'].inputs.map(function (node) { return node.name; })).toEqual([
                    'Shape', 'Value'
                ]);
            });
            it('should find the children nodes', function () {
                expect(graph.nodes['image_placeholder'].children.map(function (node) { return node.name; }))
                    .toEqual(['Conv2D']);
            });
            it('should map the input params', function () {
                expect(graph.nodes['Fill'].params['shape'].inputIndex).toEqual(0);
                expect(graph.nodes['Fill'].params['value'].inputIndex).toEqual(1);
            });
            it('should map the attribute params', function () {
                expect(graph.nodes['Conv2D'].params['strides'].value).toEqual([
                    1, 2, 2, 1
                ]);
                expect(graph.nodes['Conv2D'].params['pad'].value).toEqual('valid');
                expect(graph.nodes['Conv2D'].params['useCudnnOnGpu'].value)
                    .toEqual(true);
            });
            it('should map the placeholder attribute params', function () {
                expect(graph.nodes['image_placeholder'].params['shape'].value).toEqual([
                    3, 3, 3, 1
                ]);
                expect(graph.nodes['image_placeholder'].params['dtype'].value)
                    .toEqual('float32');
            });
            it('should map params with deprecated name', function () {
                expect(graph.nodes['Squeeze'].params['axis'].value).toEqual([1, 2]);
            });
        });
    });
});
//# sourceMappingURL=operation_mapper_test.js.map