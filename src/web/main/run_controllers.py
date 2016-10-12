__author__ = 'anushabala'
import time


def run_controllers(controller_queue):
    while True:
        if controller_queue.empty():
            time.sleep(1) # wait for some time before trying to iterate over the queue again
        else:
            # get a controller from the queue
            controller = controller_queue.get()
            # do some stuff
            controller.step()
            # todo check whether controller is done, add back only if it isn't
            controller_queue.put(controller)
