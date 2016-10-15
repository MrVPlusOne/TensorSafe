# TensorSafe

### Encode tensor/matrix shapes into types

## Motivation

Writing tensor/matrix code is error-prone. Because the compiler doesn't
 warn you when you write code to multiply two matrices with incompatible shapes.
 Instead, it lets your program run and run... oops! Suddenly crashed. I 
 can recall myself wasting quite a lot time debugging problems like this.
 Wouldn't it be nice if we can detect such errors at compiling time?
 
Here comes **TensorSafe**, a tensor computing wrapper library for
[nd4j](https://github.com/deeplearning4j/nd4j). TensorSafe encodes tensor
shape information into their types and uses type-level programming to detect
inappropriate computations and compute the resulting shapes for you.
 

## Basic Example

### Define the Dimensions

```scala
package tensorsafe.example

import tensorsafe._

trait DataNumber extends VarDim

trait DataDimension extends VarDim

trait FeatureDimension extends VarDim

```