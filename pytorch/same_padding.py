def same_padding(input, kernel_size = 3, stride= 1):
    input_size = [input.size(2), input.size(3)]
    out_size = [(input_size[0] + stride - 1)//stride,(input_size[1] + stride - 1)//stride]
    
    padding = [max(0, (out_size[0] - 1) * stride + (kernel_size - 1)+ 1 - input_size[0]),
     max(0, (out_size[1] - 1) * stride + (kernel_size - 1) + 1 - input_size[1])]
     
    is_odd = [padding[0] % 2 != 0, padding[1] % 2 != 0]
    # padding right column and bottom row if the padding is odd
    if is_odd[0] or is_odd[1]:
        input = F.pad(input ,[0, int(is_odd[1]), 0, int(is_odd[0])])
        
    return F.pad(input, [padding[1]//2, padding[1]//2, padding[0]//2, padding[0]//2])
