Para la ejecucion del EdgeDetectionFilter existen tres casos, cuya ejecucion puede ser realizada de la siguiente manera de acuerdo a cada tipo de implementacion:

* Implementacion Serial

  `g++ -o <outputfile> edgeDetSerial.c && ./<outputfile>`

* Implementacion CUDA

  `nvcc edgeDetCuda.cu -o <outputfile> && ./<outputfile>`

* Implementacion Neon ARM
  
  `g++ -o <outputfile> edgeDetNeon.c && ./<outputfile>`

