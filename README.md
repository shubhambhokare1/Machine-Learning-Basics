# Machine-Learning-Basics

Forward Pass API

API (Class: NeuralNetwork)                                   
-- create a list of matrices Θ
__init__([[int] in, [int] h1, [int] h2, …, [int] out])

-- returns Θ[layer]
[2D FloatTensor] getLayer([int] layer)

-- feedforward pass
[2D FloatTensor] forward([2D FloatTensor] input)
The input will have dimensions (batch_size, input_size). Batch size or number of input examples will be 1 for this homework.
The output will have dimensions (batch_size, output_size). output_size will be 1 as there is only 1 output from each logic gate.

Secondary API (logic_gates.py)                        
Class: [boolean] AND([boolean] x, [boolean] y)
Class: [boolean]  OR([boolean] x, [boolean] y)
Class: [boolean] NOT([boolean] x)
Class: [boolean] XOR([boolean] x, [boolean] y)
