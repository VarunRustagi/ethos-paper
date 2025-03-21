from locust import HttpUser, task, between

class MyUser(HttpUser):
    host = "http://localhost:8000"  # Explicitly define the target
    wait_time = between(1, 2)       # Optional: adds random wait between requests

    @task
    def infer(self):
        with self.client.post("/inference", json={"test": "readmission", "data": "tokenized_datasets/mimic_test_timelines_p10.hdf5", "vocab": "tokenized_datasets/mimic_vocab_t4367.pkl", "model": "/dbfs/FileStore/tables/mnt/models/best_model.pt", "output": "/dbfs/FileStore/tables/mnt/results/readmission",  "mode": "profile", "n_jobs": 8, "n_gpus": 1}, catch_response=True) as response:
            latency = response.elapsed.total_seconds()
            throughput = 1 / latency
            print(f"Latency: {latency}s, Throughput: {throughput} req/s")
