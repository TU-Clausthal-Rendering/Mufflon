#pragma once

#include "core/scene/geometry/polygon.hpp"
#include <condition_variable>
#include <future>
#include <mutex>
#include <thread>
#include <queue>

namespace mufflon::renderer::decimaters::combined {

class GpuReductionQueue {
public:
	GpuReductionQueue(const ei::UVec3 gridRes) :
		m_gridRes{ gridRes },
		m_tasks{},
		m_running{ false },
		m_exit{ false },
		m_queueMutex{},
		m_queueEmpty{},
		m_workersIdle{},
		m_thread{ [this]() { this->start(); } }
	{}
	~GpuReductionQueue() {
		m_exit = true;
		m_queueEmpty.notify_all();
		m_thread.join();
	}

	std::future<void> queue(scene::geometry::Polygons* polygon) {
		std::unique_lock lock{ m_queueMutex };

		std::promise<void> promise{};
		auto future = promise.get_future();
		m_tasks.emplace(polygon, std::move(promise));
		lock.unlock();
		m_queueEmpty.notify_all();
		return std::move(future);
	}

	// Waits until the task queue is empty
	void join() {
		std::unique_lock<std::mutex> lock(m_queueMutex);

		m_workersIdle.wait(lock, [this]() {
			return !m_running && m_tasks.empty();
		});
	}

private:
	// Starts the queue
	void start() {
		while(!m_exit) {
			std::unique_lock<std::mutex> lock(m_queueMutex);
			if(m_tasks.empty()) {
				// Sleep until task is there
				m_queueEmpty.wait(lock, [this]() { return !m_tasks.empty() || m_exit; });
				if(m_exit)
					break;
			}

			m_running = true;
			auto task = std::move(m_tasks.front());
			m_tasks.pop();

			// Unlock the queue lock - we're busy working
			lock.unlock();

			// Cluster the polygon on the GPU
			// TODO: multiple GPUs here pls
			task.first->cluster_uniformly(m_gridRes);
			task.second.set_value();

			m_running = false;
			m_workersIdle.notify_all();
		}
	}

	ei::UVec3 m_gridRes;
	std::queue<std::pair<scene::geometry::Polygons*, std::promise<void>>> m_tasks;
	std::atomic_bool m_running;
	std::atomic_bool m_exit;
	std::mutex m_queueMutex;
	std::condition_variable m_queueEmpty;
	std::condition_variable m_workersIdle;
	std::thread m_thread;
};

} // namespace mufflon::renderer::decimaters::combined