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
                cursor.execute("SELECT agent0_dialogues_evaluated, num_evals_completed FROM ActiveUsers WHERE user_id=?", (userid,))
                dialogues_evaluated, prev_num_evals = cursor.fetchone()
                updated_num_evals = prev_num_evals + 1

                # Update scenarios evaluated
                dialogues_evaluated = json.loads(dialogues_evaluated)
                dialogues_evaluated.append(results["dialogue_id"])

                cursor.execute("UPDATE ActiveUsers SET num_evals_completed=?, agent0_dialogues_evaluated=?, timestamp=? WHERE user_id=?", (updated_num_evals,
                                                                                                                                   json.dumps(dialogues_evaluated), now, userid))
            else:
                cursor.execute("SELECT agent1_dialogues_evaluated, num_evals_completed FROM ActiveUsers WHERE user_id=?", (userid,))
                dialogues_evaluated, prev_num_evals = cursor.fetchone()
                updated_num_evals = prev_num_evals + 1

                # Update scenarios evaluated
                dialogues_evaluated = json.loads(dialogues_evaluated)
                dialogues_evaluated.append(results["dialogue_id"])

                cursor.execute("UPDATE ActiveUsers SET num_evals_completed=?, agent1_dialogues_evaluated=?, timestamp=? WHERE user_id=?", (updated_num_evals,
                                                                                                                                   json.dumps(dialogues_evaluated), now, userid))

            try:
                # Update number of evals on dialogue
                cursor.execute("SELECT num_agent0_evals, num_agent1_evals, agent_mapping FROM ActiveDialogues WHERE dialogue_id=?", (results["dialogue_id"],))
                num_agent0_evals, num_agent1_evals, agent_mapping = cursor.fetchone()


                # Record answers to evaluation
                cursor.execute("INSERT INTO Responses VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)", (results["dialogue_id"], scenario_id, agent_mapping, userid, agent_id, results["humanlike"],
                                                                                      results["correct"], results["cooperative"],
                                                                                      results["cooperative_text"], results["fluent_text"]))
                if agent_id == 0:
                    num_agent0_evals += 1
                else:
                    num_agent1_evals += 1

                # Dialogue has been evaluated requisite number of times so move to CompletedDialogues
                if num_agent0_evals >= app.config["num_evals_per_dialogue"] and num_agent1_evals >= app.config["num_evals_per_dialogue"]:
                    cursor.execute("INSERT INTO CompletedDialogues VALUES (?,?,?,?,?,?)",
                                   (results["dialogue_id"], scenario_id, agent_mapping, num_agent0_evals, num_agent1_evals, now))
                    print "dialogue removed: ", results["dialogue_id"]
                    cursor.execute("DELETE FROM ActiveDialogues WHERE dialogue_id=?", (results["dialogue_id"],))
                else:
                    # Update number of evals completed for dialogue
                    cursor.execute("UPDATE ActiveDialogues SET num_agent0_evals=?, num_agent1_evals=? WHERE dialogue_id=?", (num_agent0_evals, num_agent1_evals, results["dialogue_id"]))

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


    def _check_dialogue(self, d, agent0_dialogues_evaluated, agent1_dialogues_evaluated, app):
        """
        Check if given dialogue can be presented to Turker for eval
        :param d: Dialogue to consider
        :param agent0_dialogues_evaluated: Which dialogues agent has already evaluated
        :param agent1_dialogues_evaluated: Which dialogues agent has already evaluated
        :param app: App info
        :return:
        """
        # Num agent0_evals < num evals per dialogue
        if d[6] < app.config["num_evals_per_dialogue"]:
            # Found a dialogue not previously shown to user
            if d[0] not in agent0_dialogues_evaluated and len(json.loads(d[2])) > 0:
                selected = {"agent_id": 0, "dialogue_id": d[0], "scenario_id": d[1], "events": d[2],
                            "column_names": d[3], "kb": d[4]}
                return selected
        if d[7] < app.config["num_evals_per_dialogue"]:
            if d[0] not in agent1_dialogues_evaluated and len(json.loads(d[2])) > 0:
                selected = {"agent_id": 1, "dialogue_id": d[0], "scenario_id": d[1], "events": d[2],
                            "column_names": d[3], "kb": d[5]}
                return selected

        return None


    def get_dialogue(self, userid, app):
        """
        Get an unfinished dialogue to display for user. Return necessary information
        for displaying on frontend
        :param userid:
        :return:
        """
        with self.conn:
            cursor = self.conn.cursor()

            try:
                cursor.execute("SELECT dialogue_id, scenario_id, events, column_names, agent0_kb, agent1_kb, num_agent0_evals, num_agent1_evals FROM ActiveDialogues")
                dialogues = cursor.fetchall()

                cursor.execute("SELECT agent0_dialogues_evaluated, agent1_dialogues_evaluated FROM ActiveUsers WHERE user_id=?", (userid,))
                dialogues_evaluated = cursor.fetchone()
                print dialogues_evaluated
                agent0_dialogues_evaluated = set(json.loads(dialogues_evaluated[0]))
                agent1_dialogues_evaluated = set(json.loads(dialogues_evaluated[1]))
                #print "EVALUATED: ", agent0_dialogues_evaluated, "\t", agent1_dialogues_evaluated
                selected = None
                count = 0
                while True:
                    if count >= len(dialogues) * 2:
                        for d in dialogues:
                            selected = self._check_dialogue(d, agent0_dialogues_evaluated, agent1_dialogues_evaluated, app)
                            if selected is None:
                                print "No more dialogues for current user!"
                                raise Exception
                            else:
                                return selected


                    agent_type = random.randint(0, 1)
                    d = random.sample(dialogues, 1)[0]
                    if agent_type == 0:
                        # Num agent0_evals < num evals per dialogue
                        if d[6] < app.config["num_evals_per_dialogue"]:
                            # Found a dialogue not previously shown to user
                            if d[0] not in agent0_dialogues_evaluated and len(json.loads(d[2])) > 0:
                                selected = {"agent_id": 0, "dialogue_id": d[0], "scenario_id": d[1], "events": d[2],
                                            "column_names": d[3], "kb": d[4]}
                                break
                    else:
                        if d[7] < app.config["num_evals_per_dialogue"]:
                            if d[0] not in agent1_dialogues_evaluated and len(json.loads(d[2])) > 0:
                                selected = {"agent_id": 1, "dialogue_id": d[0], "scenario_id": d[1], "events": d[2],
                                            "column_names": d[3], "kb": d[5]}
                                break
                    count += 1
                return selected
            except Exception as e:
                print "Error sampling from DB: ", e
                return None


    def close(self):
        self.conn.close()
        self.conn = None

