from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import asyncio

from livekit.agents import Agent, function_tool, RunContext, inference
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

INTRO_TIMEOUT = 180
EXP_TIMEOUT = 180
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
    supervisor: object | None = None

    intro_complete: bool = False
    experience_complete: bool = False

    intro_stage_started_at: datetime = field(default_factory=datetime.now)
    past_stage_started_at: datetime = field(default_factory=datetime.now)

    transition_to_past_requested: bool = False
    transition_to_end_requested: bool = False

    transition_in_progress: bool = False


# ============================================================
# Transition Logic
# ============================================================


@function_tool()
async def mark_intro_complete(
    context: RunContext,
) -> str:
    """
    Call when the candidate has provided
    sufficient background, skills,
    and education/work history.
    """

    logger.info(
        "\n-------------------------------\nmark_intro_complete CALLED\n-------------------------------"
    )

    state = context.session.userdata

    state.intro_complete = True

    await state.supervisor.evaluate_transition()

    return


@function_tool()
async def mark_past_complete(
    context: RunContext,
) -> str:
    """
    Call when the candidate has provided
    sufficient background, skills,
    and education/work history.
    """
    logger.info(
        "\n-------------------------------\nmark_past_complete CALLED\n-------------------------------"
    )
    state = context.session.userdata

    state.experience_complete = True

    await state.supervisor.evaluate_finish()

    return


# @function_tool()
# async def start_intro_timer(context: RunContext) -> str:
#     """
#     Call as soon as the user speaks.
#     """
#     state = context.session.userdata

#     state.intro_stage_started_at = datetime.now()


# ============================================================
# Stage Agents
# ============================================================


class SelfIntroductionAgent(Agent):
    def __init__(self):
        super().__init__(
            llm=inference.LLM(model="openai/gpt-5.2-chat-latest"),
            instructions=f"""
            GUIDELINES:

            You are responsible ONLY for the self-introduction stage.

            Goals:
            - Learn the candidate's background
            - Learn their skills
            - Learn their education or career history

            Rules:
            - Ask one question at a time.
            - Never repeat a question.
            - Ask at most {MAX_INTRO_QUESTIONS} questions.
            - Do not discuss projects or past experience.

            ON COMPLETION
            Once you have completed the goals, call mark_intro_complete()
            """,
            tools=[mark_intro_complete],
        )


class PastExperienceAgent(Agent):
    def __init__(self):
        super().__init__(
            llm=inference.LLM(model="openai/gpt-5.2-chat-latest"),
            instructions=f"""
            Transition the conversation smoothly from what has already been discussed. DO NOT REINTRODUCE THIS SECTION.
            DO NOT DO ANYTHING THAT IS NOT INSTRUCTED.
            GUIDELINES:

            You are responsible ONLY for the past experience section.

            Goals:
            - Learn about previous projects
            - Learn accomplishments
            - Learn responsibilities

            Rules:
            - Ask one question at a time.
            - Never repeat a question
            - Ask at most {MAX_EXPERIENCE_QUESTIONS} questions.
            - Do not restart the introduction stage.

            ON COMPLETEION
            Once you have accomplished the goals, call mark_past_complete()
            """,
            tools=[mark_past_complete],
        )


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

    async def evaluate_finish(self):
        state = self.state
        logger.info(
            "\n-------------------------------\nevaluate_finish CALLED\n-------------------------------"
        )

        if state.transition_in_progress:
            return

        if state.stage == InterviewStage.PAST_EXPERIENCE and state.experience_complete:
            await self.transition_to_end()

    async def transition_to_experience(self):
        async with self.lock:
            logger.info(
                "\n-------------------------------\ntransition_to_experience CALLED\n-------------------------------"
            )

            state = self.state

            if state.transition_in_progress:
                return

            logger.info(f"Transitioning from {state.stage} to PAST_EXPERIENCE")
            state.transition_in_progress = True

            try:

                logger.info(
                    f"---------------------\nGENERATING REPLY; THANKING FOR INTRO\n---------------------"
                )

                self.session.update_agent(PastExperienceAgent())

                await self.session.generate_reply(instructions="""
                    YOUR TASK NOW IS TO TRANSITION TO THE PAST EXPERIENCE STAGE FROM THE INTRODUCTION STAGE.
                    
                    GUIDE:
                    If the user is still speaking, wait for them to finish!
                    Be sure to thank the candidate for introducing themselves.

                    Briefly summarize that we now understand
                    their background.

                    Then transition naturally into discussing
                    previous work experience.
                    """)

                state.stage = InterviewStage.PAST_EXPERIENCE
                state.past_stage_started_at = datetime.now()

            finally:
                state.transition_in_progress = False

    async def transition_to_end(self):
        async with self.lock:
            logger.info(
                "\n-------------------------------\ntransition_to_end CALLED\n-------------------------------"
            )
            state = self.state

            state.experience_complete = True

            if state.transition_in_progress:
                return

            logger.info(f"Transitioning from {state.stage} to END")

            state.transition_in_progress = True

            try:
                logger.info(
                    f"---------------------\nGENERATING REPLY; THANKING FOR PAST\n---------------------"
                )

                await self.session.generate_reply(instructions="""
                    YOU ARE NOW FINISHED WITH THE PAST EXPERIENCE STAGE. YOU SHOULD NOW TRANSITION SMOOTHLY TO THE END OF THE SCREENING PROCESS.
                    DO NOT REPEAT YOURSELF

                    Finish the screening process by following the below guidelines.
                                                  
                    GUIDELINES:
                    If the user is still speaking, wait for them to finish!

                    Make sure you thank the candidate for talking about their past experience.

                    Briefly summarize that we now understand
                    their experiences.

                    Then transition naturally to the end of the conversation.
                    Be sure to say goodbye to signal the end of the screening.
                                                  
                    Mention:
                    * They will hear back within 10 business days
                    * Next steps if they are selected

                    SCREENING IS OVER NOW.
                    """)

                state.stage = InterviewStage.COMPLETE

            finally:
                state.transition_in_progress = False


# ============================================================
# Timeout Fallback
# ============================================================


async def force_transition_to_experience(supervisor):
    logger.info(
        f"---------------------\nforce_transition_to_experience CALLED\n---------------------"
    )
    state = supervisor.state

    if state.stage != InterviewStage.SELF_INTRO:
        return

    state.intro_complete = True

    await supervisor.transition_to_experience()


async def force_transition_to_end(supervisor):
    logger.info(
        f"---------------------\nforce_transition_to_end CALLED\n---------------------"
    )
    state = supervisor.state

    if state.stage != InterviewStage.PAST_EXPERIENCE:
        return

    state.experience_complete = True

    await supervisor.transition_to_end()


async def intro_timeout_monitor(supervisor):

    while True:
        state = supervisor.state

        if state.stage == InterviewStage.PAST_EXPERIENCE:
            return

        elapsed = (datetime.now() - state.intro_stage_started_at).total_seconds()
        logger.info(f"Elapsed time: {elapsed} in intro timeout monitor")
        logger.info(f"STAGE = {state.stage}; TIMEOUT = {INTRO_TIMEOUT}")

        if state.stage == InterviewStage.SELF_INTRO and elapsed >= INTRO_TIMEOUT:
            await force_transition_to_experience(supervisor)

        await asyncio.sleep(5)


async def past_timeout_monitor(supervisor):

    while True:
        state = supervisor.state

        if state.stage == InterviewStage.COMPLETE:
            return

        elapsed = (datetime.now() - state.past_stage_started_at).total_seconds()
        logger.info(f"Elapsed time: {elapsed} in past timeout monitor")
        logger.info(f"STAGE = {state.stage}; TIMEOUT = {EXP_TIMEOUT}")

        if state.stage == InterviewStage.PAST_EXPERIENCE and elapsed >= EXP_TIMEOUT:
            await force_transition_to_end(supervisor)

        await asyncio.sleep(5)


# ============================================================
# Startup
# ============================================================


async def start_interview(session):

    supervisor = session.userdata.supervisor

    session.update_agent(SelfIntroductionAgent())

    await session.generate_reply(instructions="""
        Say hello and ask the candidate to introduce themselves.
        """)

    session.userdata.intro_stage_started_at = datetime.now()
    asyncio.create_task(intro_timeout_monitor(supervisor))
    asyncio.create_task(past_timeout_monitor(supervisor))

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


async def _integration_test():
    pass


# Keeping this for testing
if __name__ == "__main__":
    asyncio.run(_integration_test())
