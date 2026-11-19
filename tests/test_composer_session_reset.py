"""The composer's session history must not leak from one compose call into the next."""

from voxposer.lmp import DEFAULT_LMP_CONFIGS, LMP, _drop_query_echo, compose_with_repair

VALID_PLAN = "ret_val = [(None, None, 'static', 'push', 'drawer_handle')]"
INVALID_PLAN = "ret_val = [(None, None, 'static', 'shove', 'drawer_handle')]"


class RecordingBackend:
    """Stands in for the LLM: returns scripted plans and records every prompt."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.prompts = []

    def generate(self, prompt, stop, image_bytes=None):
        self.prompts.append(prompt)
        return self._responses.pop(0)


def _composer(responses):
    backend = RecordingBackend(responses)
    return LMP(
        "composer", dict(DEFAULT_LMP_CONFIGS["composer"]), {}, {}, backend
    ), backend


def test_earlier_episode_does_not_appear_in_later_prompt():
    composer, backend = _composer([VALID_PLAN, VALID_PLAN])

    compose_with_repair(composer, "slide the door to the left")
    compose_with_repair(composer, "close the cabinet drawer")

    assert "slide the door to the left" not in backend.prompts[1]


def test_same_instruction_gives_the_same_prompt_every_episode():
    composer, backend = _composer([VALID_PLAN, VALID_PLAN, VALID_PLAN])

    compose_with_repair(composer, "close the cabinet drawer")
    compose_with_repair(composer, "open the drawer")
    compose_with_repair(composer, "close the cabinet drawer")

    assert backend.prompts[0] == backend.prompts[2]


def test_repair_reprompt_still_sees_the_rejected_plan():
    composer, backend = _composer([INVALID_PLAN, VALID_PLAN])

    compose_with_repair(composer, "close the cabinet drawer")

    assert len(backend.prompts) == 2
    assert "'shove'" in backend.prompts[1]


def test_the_plan_can_read_the_instruction_it_was_composed_for():
    plan = "ret_val = [(instruction, None, 'static', 'push', 'drawer_handle')]"
    composer, _ = _composer([plan, plan])

    first = compose_with_repair(
        composer, "open the drawer", scene_context="# Held block: none"
    )
    second = compose_with_repair(composer, "close the drawer")

    assert first[0][0] == "open the drawer"
    assert second[0][0] == "close the drawer"


PLACE_HELD = "ret_val = [(None, None, 'static', 'place', 'block')]"
GRASP_THEN_PLACE = "ret_val = [(None, None, 'track', 'grasp', 'red_block'), (None, None, 'static', 'place', 'red_block')]"


def test_a_plan_that_places_a_held_block_is_rejected_when_nothing_is_held():
    composer, backend = _composer([PLACE_HELD, GRASP_THEN_PLACE])

    plan = compose_with_repair(
        composer,
        "place the held block on another one",
        scene_context="# Held block: none",
    )

    assert [stage[3] for stage in plan] == ["grasp", "place"]
    repair = backend.prompts[1].split("# Query:")[-1]
    assert "place the held block on another one" in repair
    assert "Nothing is held" in repair


def test_placing_a_held_block_is_accepted_when_one_is_held():
    composer, backend = _composer([PLACE_HELD])

    plan = compose_with_repair(
        composer,
        "place the held block on another one",
        scene_context="# Held block: red_block",
    )

    assert [stage[3] for stage in plan] == ["place"]
    assert len(backend.prompts) == 1


QUERY = "# Query: the largest block on the table."
CODE = "ret_val = max(blocks('table'), key=size)"


def test_a_completion_that_repeats_the_query_keeps_its_code():
    completion = f"objects = ['red block']\n{QUERY}\n{CODE}"

    assert _drop_query_echo(completion, QUERY) == CODE


def test_a_completion_without_an_echo_is_left_alone():
    assert _drop_query_echo(CODE, QUERY) == CODE


def test_a_later_example_is_not_mistaken_for_an_echo():
    completion = (
        f"{CODE}\n\n# Query: something else.\nret_val = 1\n{QUERY}\nret_val = 2"
    )

    assert _drop_query_echo(completion, QUERY) == completion
