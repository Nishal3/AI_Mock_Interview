from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import asyncio

from livekit.agents import Agent
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

INTRO_TIMEOUT = 180
MAX_INTRO_QUESTIONS = 3
MAX_EXPERIENCE_QUESTIONS = 4


# ============================================================
# Interview State
# ============================================================


class InterviewStage(str, Enum):
    SELF_INTRO = "self_intro"
    PAST_EXPERIENCE = "past_experience"
    COMPLETE = "complete"


@dataclass
class InterviewState:
    stage: InterviewStage = InterviewStage.SELF_INTRO

    intro_questions_asked: int = 0
    experience_questions_asked: int = 0

    intro_complete: bool = False
    experience_complete: bool = False

    stage_started_at: datetime = field(default_factory=datetime.now)

    transition_in_progress: bool = False

    asked_questions: set[str] = field(default_factory=set)


# ============================================================
# Stage Agents
# ============================================================


class SelfIntroductionAgent(Agent):
    def __init__(self):
        super().__init__(instructions="""
            You are responsible ONLY for the self-introduction stage.

            Goals:
            - Learn the candidate's background
            - Learn their skills
            - Learn their education or career history

            Rules:
            - Ask one question at a time.
            - Never repeat a question.
            - Ask at most 3 questions.
            - Do not discuss projects or past experience.
            """)


class PastExperienceAgent(Agent):
    def __init__(self):
        super().__init__(instructions="""
            You are responsible ONLY for the past-experience stage.

            Goals:
            - Learn about previous projects
            - Learn accomplishments
            - Learn responsibilities

            Rules:
            - Ask one question at a time.
            - Ask at most 4 questions.
            - Do not restart the introduction stage.
            """)


# ============================================================
# Supervisor
# ============================================================


class InterviewSupervisor:

    def __init__(self, session):
        self.session = session
        self.lock = asyncio.Lock()

        if not hasattr(session, "userdata") or session.userdata is None:
            session.userdata = InterviewState()

    @property
    def state(self) -> InterviewState:
        return self.session.userdata

    async def evaluate_transition(self):

        state = self.state

        if state.transition_in_progress:
            return

        if state.stage == InterviewStage.SELF_INTRO and state.intro_complete:
            await self.transition_to_experience()

    async def transition_to_experience(self):
        async with self.lock:

            state = self.state

            if state.transition_in_progress:
                return

            logger.info(f"Transitioning from {state.stage} to PAST_EXPERIENCE")

            state.transition_in_progress = True

            try:

                await self.session.generate_reply(instructions="""
                    Thank the candidate for the introduction.

                    Briefly summarize that we now understand
                    their background.

                    Then transition naturally into discussing
                    previous work experience.
                    """)

                state.stage = InterviewStage.PAST_EXPERIENCE
                state.stage_started_at = datetime.now()

                self.session.update_agent(PastExperienceAgent())

            finally:
                state.transition_in_progress = False


# ============================================================
# Completion Logic
# ============================================================


def update_intro_completion(state: InterviewState):

    if state.intro_questions_asked >= MAX_INTRO_QUESTIONS:
        state.intro_complete = True


def update_experience_completion(state: InterviewState):

    if state.experience_questions_asked >= MAX_EXPERIENCE_QUESTIONS:
        state.experience_complete = True
        state.stage = InterviewStage.COMPLETE


# ============================================================
# Timeout Fallback
# ============================================================


async def force_transition_to_experience(supervisor):

    state = supervisor.state

    if state.stage != InterviewStage.SELF_INTRO:
        return

    await supervisor.transition_to_experience()


async def timeout_monitor(supervisor):

    while True:

        state = supervisor.state

        elapsed = (datetime.now() - state.stage_started_at).total_seconds()
        logger.info(f"Elapsed time: {elapsed}")

        if state.stage == InterviewStage.SELF_INTRO and elapsed >= INTRO_TIMEOUT:
            await force_transition_to_experience(supervisor)

        await asyncio.sleep(5)


# ============================================================
# Hooks
# ============================================================


async def on_intro_question_asked(supervisor):

    state = supervisor.state

    state.intro_questions_asked += 1

    update_intro_completion(state)

    await supervisor.evaluate_transition()


async def on_experience_question_asked(supervisor):

    state = supervisor.state

    state.experience_questions_asked += 1

    update_experience_completion(state)


# ============================================================
# Startup
# ============================================================


async def start_interview(session):

    session.userdata = InterviewState()

    supervisor = InterviewSupervisor(session)

    session.update_agent(SelfIntroductionAgent())

    asyncio.create_task(timeout_monitor(supervisor))

    await session.generate_reply(instructions="""
        Begin the interview.

        Ask the candidate to introduce themselves.
        """)

    return supervisor


# ============================================================
# Testing
# ============================================================


class _MockSessionNoAPI:

    def __init__(self):
        self.userdata = None
        self.current_agent = None

    async def generate_reply(self, instructions):
        print("\n=== AGENT RESPONSE ===")
        print(instructions)

    def update_agent(self, agent):
        self.current_agent = agent

        print(f"\n[AGENT SWITCHED]" f" -> {agent.__class__.__name__}")


class _MockSessionNoAPI:

    def __init__(self):
        self.userdata = None
        self.current_agent = None

    async def generate_reply(self, instructions):
        print("\n=== AGENT RESPONSE ===")
        print(instructions)

    def update_agent(self, agent):
        self.current_agent = agent

        print(f"\n[AGENT SWITCHED]" f" -> {agent.__class__.__name__}")


async def _integration_test():

    print("\n==========")
    print("TEST START")
    print("==========")
    session = _MockSessionNoAPI()

    supervisor = await start_interview(session)

    print("\nInitial State:")
    print(session.userdata)

    assert session.userdata.stage == InterviewStage.SELF_INTRO

    print("\n----------")
    print("INTRO QUESTIONS")
    print("----------")

    await on_intro_question_asked(supervisor)

    await on_intro_question_asked(supervisor)

    await on_intro_question_asked(supervisor)

    print("\nState after introduction:")
    print(session.userdata)

    assert session.userdata.intro_complete

    print(f"Intro Complete: " f"{session.userdata.intro_complete}")

    await supervisor.evaluate_transition()

    assert session.userdata.stage == InterviewStage.PAST_EXPERIENCE

    print(f"Current Stage: " f"{session.userdata.stage}")

    print("\n----------")
    print("EXPERIENCE QUESTIONS")
    print("----------")

    await on_experience_question_asked(supervisor)
    await on_experience_question_asked(supervisor)
    await on_experience_question_asked(supervisor)
    await on_experience_question_asked(supervisor)

    print("\nState after past experience:")
    print(session.userdata)

    assert session.userdata.stage == InterviewStage.COMPLETE

    print(f"Current Stage: " f"{session.userdata.stage}")

    print("\n==========")
    print("TEST PASSED")
    print("==========")


# Keeping this for testing
if __name__ == "__main__":
    asyncio.run(_integration_test())
