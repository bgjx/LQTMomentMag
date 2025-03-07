layer_boundaries = "-3.00,-1.90; -1.90,-0.59; -0.59, 0.22; 0.22, 2.50; 2.50, 7.00; 7.00,9.00;  9.00,15.00 ; 15.00,33.00; 33.00,9999"

layer_boundary = [[float(v) for v in layer.split(",")] for layer in layer_boundaries.split(";")]
print(layer_boundary)