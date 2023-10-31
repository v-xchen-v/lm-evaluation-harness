"""
a background task queue, can add new task, list new added tasks, pending tasks, and finished tasks, and make sure only one task is processed at a time.

use .pkl file to serialize and recovery task queue
"""
import queue
import threading
import time
import pickle
from pathlib import Path
from autoeval_server.evalrunner.runeval import run_eval

verbose = False
class TaskQueue:
    def __init__(self, filename=None):
        self.filename = filename
        if self.filename is None or not Path(self.filename).is_file():
            self.task_queue = queue.Queue()
            self.finished_tasks = []

        else:
            self.recover_tasks() 

        self.processing_task = None
        self.worker_thread = threading.Thread(target=self._process_tasks)
        self.worker_thread.daemon = True
        self.worker_thread.start()


    def _process_tasks(self):
        while True:
            if self.task_queue.empty():
                time.sleep(1)
                if verbose:
                    print("waiting for task request...")
            else:
                task = self.task_queue.get()
                self.processing_task = task
                print(f"Processing task: '{task}'")
                # record task status before processing
                self.serialize_tasks()

                self._process_task(task)

                print(f"Task '{task}' finished.")
                self.finished_tasks.append(task)
                self.processing_task = None
                self.task_queue.task_done()
                # record task status after processing
                self.serialize_tasks()

    def _process_task(self, task):
        # add your task processing logic here
        try:
            run_eval(task)
        except EnvironmentError as e: # wrong name of huggingface model
            print(e)
        except Exception as e:
            print("Evaluation cancelled by unhandled exception:")
            print(e)
        
    def list_pending_tasks(self):
        if self.task_queue is None:
            return []
        return list(self.task_queue.queue)
    
    def add_task(self, task):
        if task not in self.task_queue.queue and task != self.processing_task:
            self.task_queue.put(task)
            print(f"Task '{task}' added to the queue")
        else:
            print(f"Task '{task}' already there! Skip.")

    def list_tasks(self):
        print("-------- Pending Tasks --------")
        for task in self.list_pending_tasks():
            print(task)
        print("-------- Processing Task --------")
        if self.processing_task:
            print(self.processing_task)
        print("-------- Finished Tasks --------")
        for task in self.finished_tasks:
            print(task)
        return self.list_pending_tasks(), [] if self.processing_task is None else [self.processing_task], self.finished_tasks
    
    def serialize_tasks(self):
        tasks = {
            'pending_tasks': self.list_pending_tasks() + ([] if self.processing_task is None else [self.processing_task]),
            'finished_tasks': self.finished_tasks
        }
        try:
            with open(self.filename, 'wb') as file:
                pickle.dump(tasks, file)
        except FileNotFoundError:
            print("warning: not cache file specified.")

    def recover_tasks(self):
        try:
            with open(self.filename, 'rb') as file:
                tasks = pickle.load(file)
                self.finished_tasks = tasks['finished_tasks']
                self.task_queue = queue.Queue()
                for task in tasks['pending_tasks']:
                    self.task_queue.put(task)

        except FileNotFoundError:
            print("no recovery file.") 

TASKS_PKL = Path(__file__).parent.parent /'.tasks.pkl'
task_queue = TaskQueue(TASKS_PKL)

def main():
    task_queue = TaskQueue('.tasks.pkl')

    while True:
        print("\n1. Add New Task")
        print("2. List All Tasks")
        print("3. Exit")

        choice = input("Enter your choice (1/2/3/4): ")

        if choice == "1":
            task = input("Enter the task: ")
            task_queue.add_task(task)
        elif choice == "2":
            task_queue.list_tasks()
        elif choice == "3":
            print("Exiting the program.")
            break
        else:
            print("Invalid choice. Please try again.")


if __name__ == "__main__":
    main()