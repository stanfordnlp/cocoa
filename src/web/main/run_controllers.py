__author__ = 'anushabala'
import time


def run_controllers(controller_queue):
    while True:
        if controller_queue.empty():
            time.sleep(1)  # wait for some time before trying to iterate over the queue again
        else:
            # get a controller from the queue
            controller = controller_queue.get()
            print "Got controller object"
            print "Calling controller step"
            controller.step()
            if not controller.game_over():
                print "Controller still active, adding controller back"
                controller_queue.put(controller)
            else:
                print "Controller inactive!! removing it from queue"
                print controller.sessions


def run_single_controller(controller):
    while not controller.game_over():
        controller.step()
        time.sleep(1)
