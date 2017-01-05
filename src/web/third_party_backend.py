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
            cursor.execute("INSERT OR IGNORE INTO ActiveUsers VALUES (?,?,?,?,?)",
                           (userid, json.dumps([]), json.dumps([]), 0, now))


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

            agent_id = int(results["agent_id"])

            # Update ActiveUsers to num_tasks and timestamp
            if agent_id == 0:
                cursor.execute("SELECT agent0_scenarios_evaluated, num_evals_completed FROM ActiveUsers WHERE user_id=?", (userid,))
                scenarios_evaluated, prev_num_evals = cursor.fetchone()
                updated_num_evals = prev_num_evals + 1

                # Update scenarios evaluated
                scenarios_evaluated = json.loads(scenarios_evaluated)
                scenarios_evaluated.append(scenario_id)

                cursor.execute("UPDATE ActiveUsers SET num_evals_completed=?, agent0_scenarios_evaluated=?, timestamp=? WHERE user_id=?", (updated_num_evals,
                                                                                                                                   json.dumps(scenarios_evaluated), now, userid))
            else:
                cursor.execute("SELECT agent1_scenarios_evaluated, num_evals_completed FROM ActiveUsers WHERE user_id=?", (userid,))
                scenarios_evaluated, prev_num_evals = cursor.fetchone()
                updated_num_evals = prev_num_evals + 1

                # Update scenarios evaluated
                scenarios_evaluated = json.loads(scenarios_evaluated)
                scenarios_evaluated.append(scenario_id)

                cursor.execute("UPDATE ActiveUsers SET num_evals_completed=?, agent1_scenarios_evaluated=?, timestamp=? WHERE user_id=?", (updated_num_evals,
                                                                                                                                   json.dumps(scenarios_evaluated), now, userid))

            # Record answers to evaluation
            print "AGENT ID BEFORE RESPONSES: ", agent_id
            cursor.execute("INSERT INTO Responses VALUES (?,?,?,?,?,?,?)", (scenario_id, userid, agent_id, results["humanlike"],
                                                                                  results["correct"], results["strategic"],
                                                                                  results["fluent"]))
            try:
                # Update number of evals on dialogue
                cursor.execute("SELECT num_agent0_evals, num_agent1_evals FROM ActiveDialogues WHERE scenario_id=?", (scenario_id,))
                num_agent0_evals, num_agent1_evals = cursor.fetchone()
                if agent_id == 0:
                    num_agent0_evals += 1
                else:
                    num_agent1_evals += 1

                # Dialogue has been evaluated requisite number of times so move to CompletedDialogues
                if num_agent0_evals == app.config["num_evals_per_dialogue"] and num_agent1_evals == app.config["num_evals_per_dialogue"]:
                    cursor.execute("INSERT INTO CompletedDialogues VALUES (?,?,?,?)",
                                   (scenario_id, num_agent0_evals, num_agent1_evals, now))
                    cursor.execute("DELETE FROM ActiveDialogues WHERE scenario_id=?", (scenario_id,))
                else:
                    # Update number of evals completed for dialogue
                    cursor.execute("UPDATE ActiveDialogues SET num_agent0_evals=?, num_agent1_evals=? WHERE scenario_id=?", (num_agent0_evals, num_agent1_evals, scenario_id))

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

                cursor.execute("SELECT agent0_scenarios_evaluated, agent1_scenarios_evaluated FROM ActiveUsers WHERE user_id=?", (userid,))
                scenarios_evaluated = cursor.fetchone()
                print scenarios_evaluated
                agent0_scenarios_evaluated = set(json.loads(scenarios_evaluated[0]))
                agent1_scenarios_evaluated = set(json.loads(scenarios_evaluated[1]))
                print "EVALUATED: ", agent0_scenarios_evaluated, "\t", agent1_scenarios_evaluated
                selected = None
                # TODO: Whether to change to random selection
                for d in dialogues:
                    # Found a dialogue not previously shown to user
                    if d[0] not in agent0_scenarios_evaluated and len(json.loads(d[1])) > 0:
                        print "AGENT 0!"
                        selected = {"agent_id": 0, "uuid": d[0], "events": d[1],
                                    "column_names": d[2], "kb": d[3]}
                        break
                    if d[0] not in agent1_scenarios_evaluated and len(json.loads(d[1])) > 0:
                        print "AGENT 1!"
                        selected = {"agent_id": 1, "uuid": d[0], "events": d[1],
                                    "column_names": d[2], "kb": d[4]}
                        break
                return selected
            except Exception as e:
                print "Error sampling from DB: ", e
                return None


    def close(self):
        self.conn.close()
        self.conn = None

