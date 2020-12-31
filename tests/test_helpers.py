import unittest
import covid19sim.frozen.message_utils as mu
from tests.helpers import (
    MessagesContext,
    ObservedRisk,
)


class TestMessagesContext(unittest.TestCase):

    def setUp(self):
        self.context = MessagesContext()
        self.maxDiff = None

    def test_compile_message_history_smoke_test(self):
        # test arguments
        uid = 10
        encounter_risk_level = 14
        update_time = 30
        update_risk_level = 3
        # record an encounter observation
        o = ObservedRisk(
            encounter_time=0,
            encounter_risk_level=encounter_risk_level
        )
        # append an update on the previous observation
        o.update(update_time=update_time, update_risk_level=update_risk_level)
        # compile the previous object into message history
        observed_risks = [o]
        curated_messages = self.context.compile_message_history(
            observed_risks, uid=uid
        )
        # define expected messages
        expected_encounter_message = mu.EncounterMessage(
            uid=uid,
            encounter_time=0,
            risk_level=encounter_risk_level,
            _real_encounter_time=0,
            _sender_uid=0,
            _exposition_event=False
        )
        expected_update_message = mu.create_update_message(
            encounter_message=expected_encounter_message,
            current_time=update_time,
            new_risk_level=update_risk_level
        )
        self.assertEqual(curated_messages,
                         [expected_encounter_message, expected_update_message])

    def test_generate_random_messages_smoke_test(self):
        # test arguments
        exposure_timestamp = 7
        n_encounters = 10
        n_messages = 20
        max_timestamp = 30
        # create random messages
        curated_messages = self.context.generate_random_messages(
            max_timestamp=max_timestamp,
            exposure_timestamp=exposure_timestamp,
            n_encounters=n_encounters,
            n_messages=n_messages
        )
        # test if the number of messages match up the arguments
        returned_n_encounters = sum([
            isinstance(message, mu.EncounterMessage)
            for message in curated_messages
        ])
        self.assertEqual(returned_n_encounters, n_encounters)
        self.assertEqual(len(curated_messages), n_messages)

    def test_generate_linear_saturation_risk_messages_smoke_test(self):
        # test arguments
        exposure_timestamp = 10
        n_encounters = 20
        max_timestamp = 30
        init_risk_level = 5
        final_risk_level = 12
        # create linear saturation messages
        curated_messages = self.context. \
            generate_linear_saturation_risk_messages(
                max_timestamp=max_timestamp,
                exposure_timestamp=exposure_timestamp,
                n_encounters=n_encounters,
                init_risk_level=init_risk_level,
                final_risk_level=final_risk_level
            )
        # test if the update message risks are increasing
        for message in curated_messages:
            if isinstance(message, mu.UpdateMessage):
                self.assertTrue(message.old_risk_level <=
                                message.new_risk_level)

    def tearDown(self):
        pass
