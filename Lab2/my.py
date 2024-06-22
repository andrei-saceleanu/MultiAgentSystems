from environment import *
import time

class MyAgent(BlocksWorldAgent):

    def __init__(self, name: str, desired_state: BlocksWorld):
        super(MyAgent, self).__init__(name=name)

        self.desired_state = desired_state

        self.subgoals = self.desired_state.get_stacks()
        self.current_goal_stack = None
        self.belief = None
        self.actions = []
        self.act_idx = 0


    def response(self, perception: BlocksWorldPerception):
        # TODO: revise beliefs; if necessary, make a plan; return an action.
        self.belief = perception.current_world
        if self.act_idx >= len(self.actions):
            if not self.subgoals:
                return AgentCompleted()
            self.current_goal_stack = self.subgoals.pop()
            self.actions = self.plan()
            self.act_idx = 0
        act = self.actions[self.act_idx]
        self.act_idx += 1
        return act



    def revise_beliefs(self, perceived_world_state: BlocksWorld):
        # TODO: check if what the agent knows corresponds to what the agent sees
        pass


    def plan(self) -> List[BlocksWorldAction]:
        # TODO: return a new plan, as a sequence of `BlocksWorldAction' instances, based on the agent's knowledge.
        d = {}
        plan = []
        for block in self.current_goal_stack.get_blocks():
            st_block = self.belief.get_stack(block)
            if st_block not in d:
                d[st_block] = [block]
            else:
                d[st_block].append(block)
        for st in d:
            cnt = len(d[st])
            blocks = st.get_blocks()[::-1]
            for i, block in enumerate(blocks):
                if i < len(blocks) - 1:
                    plan.extend([Unstack(block, blocks[i+1]), PutDown(block)])
                if block in d[st]:
                    cnt -= 1
                if cnt == 0:
                    break
        blocks = self.current_goal_stack.get_blocks()
        for i, block in enumerate(blocks):
            if i==0:
                plan.append(Lock(block))
            else:
                plan.extend([PickUp(block), Stack(block, blocks[i-1]), Lock(block)])
        return plan
            
    def status_string(self):
        return str(self) + f" : {[str(act) for act in self.actions[self.act_idx:]]}"



class Tester(object):
    STEP_DELAY = 0.1
    TEST_SUITE = "tests/0e-large/"

    EXT = ".txt"
    SI  = "si"
    SF  = "sf"

    DYNAMICITY = .0

    AGENT_NAME = "*A"

    def __init__(self):
        self._environment = None
        self._agents = []

        self._initialize_environment(Tester.TEST_SUITE)
        self._initialize_agents(Tester.TEST_SUITE)



    def _initialize_environment(self, test_suite: str) -> None:
        filename = test_suite + Tester.SI + Tester.EXT

        with open(filename) as input_stream:
            self._environment = DynamicEnvironment(BlocksWorld(input_stream=input_stream))


    def _initialize_agents(self, test_suite: str) -> None:
        filename = test_suite + Tester.SF + Tester.EXT

        agent_states = {}

        with open(filename) as input_stream:
            desires = BlocksWorld(input_stream=input_stream)
            agent = MyAgent(Tester.AGENT_NAME, desires)

            agent_states[agent] = desires
            self._agents.append(agent)

            self._environment.add_agent(agent, desires, None)

            print("Agent %s desires:" % str(agent))
            print(str(desires))


    def make_steps(self):
        print("\n\n================================================= INITIAL STATE:")
        print(str(self._environment))
        print("\n\n=================================================")

        completed = False
        nr_steps = 0

        while not completed:
            completed = self._environment.step()

            time.sleep(Tester.STEP_DELAY)
            print(str(self._environment))

            for ag in self._agents:
                print(ag.status_string())

            nr_steps += 1

            print("\n\n================================================= STEP %i completed." % nr_steps)

        print("\n\n================================================= ALL STEPS COMPLETED")





if __name__ == "__main__":
    tester = Tester()
    tester.make_steps()