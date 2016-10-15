# TensorSafe

### Encode tensor/matrix shapes into types

## Motivation

Writing tensor/matrix code is error-prone. Because the compiler doesn't
 warn you if you multiply two matrices with incompatible shapes in your 
 code. Instead, it lets your program run and run... oops, suddenly crashed. I 
 can recall myself wasting quite a lot time debugging problems like this.
 Wouldn't it be nice if we can detect such errors at compiling time?
 
Here comes **TensorSafe**, a tensor computing wrapper library for
[nd4j](https://github.com/deeplearning4j/nd4j). TensorSafe encodes tensor
shape information into its type and uses type-level programming to detect
inappropriate computations and compute the resulting shapes. 