def matrix_shape(matrix):
    
    res = []
    block = matrix
    while True:
        res.append(len(block))
        block = block[0]
        if type(block) is not list:
            break
    return res
