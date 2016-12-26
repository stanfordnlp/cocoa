import json
import random
import sqlite3
import time
import uuid


def get_timestamp():
    return time.strftime("%c")


class BackendConnection(object):

    def __init__(self, db_path):
        self.conn = sqlite3.connect(db_path)


    def get_num_evals_completed(self, userid):
        """
        Get number of evals current user has completed
        :param userid:
        :return:
        """
        with self.conn:
            c = self.conn.cursor()
            c.execute("SELECT num_evals_completed FROM ActiveUsers WHERE user_id=?", (userid,))
            num_evals = c.fetchone()[0]
            return num_evals


    def create_user_if_necessary(self, userid):
        """
        Adds a new user to the database if necessary
        :param :
        :return:
        """
        with self.conn:
            cursor = self.conn.cursor()
            now = get_timestamp()
            cursor.execute("INSERT OR IGNORE INTO ActiveUsers VALUES (?,?,?,?)",
                           (userid, json.dumps([]), 0, now))


    def submit_task(self, userid, scenario_id, results, app):
        """
        Submit a single task and update DB appropriately.
        :param userid: ID of user currently submitting task
        :param scenario_id: ID of scenario being submitted
        :param results: Dict with answers to evaluation questions
        :param app: Storing config details
        """
        with self.conn:
            cursor = self.conn.cursor()
            now = get_timestamp()

            # Update ActiveUsers to num_tasks and timestamp
            cursor.execute("SELECT scenarios_evaluated, num_evals_completed FROM ActiveUsers WHERE user_id=?", (userid,))
            scenarios_evaluated, prev_num_evals = cursor.fetchone()
            updated_num_evals = prev_num_evals + 1

            # Update scenarios evaluated
            scenarios_evaluated = json.loads(scenarios_evaluated)
            scenarios_evaluated.append(scenario_id)

            cursor.execute("UPDATE ActiveUsers SET num_evals_completed=?, scenarios_evaluated=?, timestamp=? WHERE user_id=?", (updated_num_evals,
                                                                                                                               json.dumps(scenarios_evaluated), now, userid))

            # Record answers to evaluation
            cursor.execute("INSERT INTO Responses VALUES (?,?,?,?,?,?,?,?,?,?)", (scenario_id, userid, results["humanlike_0"],
                                                                                  results["correct_0"], results["strategic_0"],
                                                                                  results["fluent_0"], results["humanlike_1"],
                                                                                  results["correct_1"], results["strategic_1"],
                                                                                  results["fluent_1"]))
            try:
                # Update number of evals on dialogue
                cursor.execute("SELECT num_evals FROM ActiveDialogues WHERE scenario_id=?", (scenario_id,))
                num_evals = cursor.fetchone()[0]
                updated_num_evals = num_evals + 1
                # Dialogue has been evaluated requisite number of times so move to CompletedDialogues
                if updated_num_evals == app.config["num_evals_per_dialogue"]:
                    cursor.execute("INSERT INTO CompletedDialogues VALUES (?,?,?)",
                                   (scenario_id, updated_num_evals, now))
                    cursor.execute("DELETE FROM ActiveDialogues WHERE scenario_id=?", (scenario_id,))
                else:
                    # Update number of evals completed for dialogue
                    cursor.execute("UPDATE ActiveDialogues SET num_evals=? WHERE scenario_id=?", (updated_num_evals, scenario_id))

            except TypeError as e:
                print "Catching error: ", e


    def get_finished_info(self, userid):
        """
        Get info related to a user once they have finished requisite number of tasks
        :param userid:
        :param from_mturk:
        :return:
        """
        mturk_code = "MTURK_TASK_{}".format(str(uuid.uuid4().hex))
        with self.conn:
            cursor = self.conn.cursor()
            now = get_timestamp()
            cursor.execute("SELECT num_evals_completed FROM ActiveUsers WHERE user_id=?", (userid,))
            num_evals = cursor.fetchone()[0]
            cursor.execute("INSERT INTO CompletedUsers VALUES (?,?,?,?)", (userid, mturk_code, now, num_evals))
            # Reset number of completed tasks for given user to 0
            cursor.execute("UPDATE ActiveUsers SET num_evals_completed=?, timestamp=? WHERE user_id= ?", (0, now, userid))
            return mturk_code


    def get_dialogue(self, userid):
        """
        Get an unfinished dialogue to display for user. Return necessary information
        for displaying on frontend
        :param userid:
        :return:
        """
        with self.conn:
            cursor = self.conn.cursor()

            try:
                cursor.execute("SELECT scenario_id, events, column_names, agent0_kb, agent1_kb FROM ActiveDialogues")
                dialogues = cursor.fetchall()

                cursor.execute("SELECT scenarios_evaluated FROM ActiveUsers WHERE user_id=?", (userid,))
                scenarios_evaluated = cursor.fetchone()
                print scenarios_evaluated
                scenarios_evaluated = set(json.loads(scenarios_evaluated[0]))
                print "EVAL: ", scenarios_evaluated
                selected = None
                for d in dialogues:
                    # Found a dialogue not previously shown to user
                    if d[0] not in scenarios_evaluated and len(json.loads(d[1])) > 0:
                        selected = d
                        break
                #print "Selected: ", selected
                return selected
            except Exception as e:
                print "Error sampling from DB: ", e
                return None


    def close(self):
        self.conn.close()
        self.conn = None

