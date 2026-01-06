import ctypes

import warp as wp

def create_cuda_graph_callback(callback, device=None, stream=None):
    with wp.ScopedCapture(device=device, stream=stream) as capture:
        callback()

    graph = capture.graph

    if stream is not None:
        if stream.device != graph.device:
            raise RuntimeError(f"Cannot launch graph from device {graph.device} on stream from device {stream.device}")
        device = stream.device
    else:
        device = graph.device
        stream = device.stream
    
    # populate graph executable
    if graph.graph_exec is None:
        g = ctypes.c_void_p()
        result = wp._src.context.runtime.core.wp_cuda_graph_create_exec(
            graph.device.context, stream.cuda_stream, graph.graph, ctypes.byref(g)
        )
        if not result:
            raise RuntimeError(f"Graph creation error: {wp.context.runtime.get_error_string()}")
        graph.graph_exec = g

    def graph_callback():
        wp.capture_launch(graph)

    return graph_callback
