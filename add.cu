class Add {
	
	public:

	Add(int _a, int _b, cudaStream_t _stream = 0) : 
		a(_a),
        b(_b),
        stream(_stream) {};

	int op();

	private:
	int a, b;
	cudaStream_t stream;
};

__global__ void add(int a, int b, int *c) {
 		
 		*c = a + b;
}

int Add::op() {

	int c;
	int *dev_c;

	cudaMalloc((void**)&dev_c, sizeof(int));
	add<<<1,1>>>(a,b,dev_c);
	cudaMemcpyAsync(&c, dev_c, sizeof(int), cudaMemcpyDeviceToHost, stream);
	return c;

}

