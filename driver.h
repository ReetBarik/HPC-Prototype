#include <vector>
#include "add.cu"

// class Worker {
// 	virtual ~Worker() = default;
// 	virtual int exec() = 0;
// };

class Cpu_worker {

public:

	Cpu_worker(int a, int b){
		this->a = a; 
		this->b = b; 
	}

	int exec() {
		return a + b;
	}

private:
	int a, b;
};

class Gpu_worker {

public:

	Gpu_worker(int a, int b){
		this->a = a; 
		this->b = b; 
		auto e = cudaStreamCreate(&cuda_stream_);
		this-> A = new Add(a, b);
	}

	int exec() {
		int c = A->op();
		cudaStreamSynchronize(cuda_stream_);
		return c;
	}


private:
	int a, b;
	Add *A;
	cudaStream_t cuda_stream_; 
};

class Engine {

public:

	Engine(int s, int e, int *a1, int *a2, int *a3, int cw, int gw = 0) {
		this->start = s;
		this->end = e;
		this->a1 = a1;
		this->a2 = a2;
		this->a3 = a3;
		cpu_workers_.resize(cw);
		gpu_workers_.resize(gw);
		// std::cout << "Start: " << start << " End: " << end << std::endl;
		// std::cout << "cw: " << cpu_workers_.size() << " gw: " << gpu_workers_.size() << std::endl;
		#pragma omp parallel num_threads(cw + gw)
		{
			int rank = omp_get_thread_num();
			if (rank < cw) {
				auto w = new Cpu_worker(a1[rank + start], a2[rank + start]);
				cpu_workers_[rank] = w;
			} else {
				auto w = new Gpu_worker(a1[rank + start], a2[rank + start]);
				gpu_workers_[rank - cw] = w;
			}
			
		}
	}

	void add() {

		int  threads = cpu_workers_.size() + gpu_workers_.size();
		// std::cout << "Workers: " << threads << std::endl; 
		#pragma omp parallel num_threads(threads)
		{
			int rank = omp_get_thread_num();
			if (rank < cpu_workers_.size()) {
				a3[rank + start] = cpu_workers_[rank]->exec();
			} else {
				a3[rank + start] = gpu_workers_[rank - cpu_workers_.size()]->exec();
			}
		}
		 	
	}

private:
	int start;
	int end;
	int *a1;
	int *a2;
	int *a3;
	std::vector<Cpu_worker *> cpu_workers_;
	std::vector<Gpu_worker *> gpu_workers_;
};



void driver(int *a1, int *a2, int *a3, int size, int d1, int gw) {

	int batch = 5; 
	int num_batch = size / batch;
	// std::cout << "Number of Batches: " << num_batch << std::endl;
	std::vector<Engine*> E;
	E.reserve(num_batch);

  	for (int i = 0; i < num_batch; ++i) {

  		int start = i * batch; 
		int end = start + batch;
		// std::cout << "Start: " << start << " End: " << end << std::endl;
    	auto e = new Engine(start, end, a1, a2, a3, batch - gw, gw);
    	E[i] = e;
  }

	omp_set_nested(1);

	#pragma omp parallel for num_threads(d1)
	for (int i = 0; i < num_batch; i++) {		
		E[i]->add();
	}

	
}