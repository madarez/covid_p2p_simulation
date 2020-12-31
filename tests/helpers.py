import dataclasses
import numpy as np
import typing

import covid19sim.frozen.message_utils as mu


def chronological_key(message):
    if isinstance(message, mu.UpdateMessage):
        return (message.update_time, 1)
    else:
        return (message.encounter_time, 0)


@dataclasses.dataclass
class ObservedRisk:

    @dataclasses.dataclass
    class UpdateSignature:
        update_time: mu.TimestampType
        update_risk_level: mu.RiskLevelType
        update_reason: typing.Optional[str] = None

    encounter_time: mu.TimestampType
    encounter_risk_level: mu.RiskLevelType
    update_signatures: typing.List[UpdateSignature] = \
        dataclasses.field(default_factory=list)

    def update(self,
               update_time: mu.TimestampType,
               update_risk_level: mu.RiskLevelType,
               update_reason: typing.Optional[str] = None,
               ):
        assert update_time >= self.encounter_time, \
            'update time {} < encounter time {}'.format(
                update_time,
                self.encounter_time
            )
        if len(self.update_signatures) > 0:
            assert update_time >= self.update_signatures[-1].update_time
        self.update_signatures.append(
            ObservedRisk.UpdateSignature(
                update_time, update_risk_level, update_reason
            ))


class MessagesContext():

    def __init__(self):
        self.real_uid_counter: mu.RealUserIDType = 0

    @staticmethod
    def _rotate_uid(uid: mu.UIDType, shift: int):
        if shift <= 0:
            return uid
        elif shift >= mu.message_uid_bit_count:
            return mu.create_new_uid()
        else:
            for _ in range(shift):
                uid = mu.update_uid(uid)
            return uid

    def compile_message_history(
        self,
        observed_risks: typing.List[ObservedRisk],
        uid: typing.Optional[mu.UIDType] = None,
        exposure_timestamp: mu.TimestampType = np.inf,
    ) -> typing.List[mu.GenericMessageType]:
        """
        converting observed risk levels for a new contacted user
        into a list of encounter and update messages
        :param observed_risks: a mapping from encountered time to the
        observed data at that time this data itself is a data class holding
        a mandatory field risk_level and optional lists of update signatures,
        i.e., update_risk_levels, update_times, and update_reasons for
        update messages to be related to the encountered message.
        :param uid: uid to be initialized at timestamp=0, random if not provided
        :return: user_messages
        """
        user_messages = []
        prev_time: int = 0
        if uid is None:
            uid = mu.create_new_uid()
        for observed_risk in observed_risks:
            uid = MessagesContext._rotate_uid(
                uid, observed_risk.encounter_time - prev_time)
            assert observed_risk.encounter_risk_level <= mu.message_uid_mask

            encounter_message = mu.EncounterMessage(
                uid=uid,
                risk_level=observed_risk.encounter_risk_level,
                encounter_time=observed_risk.encounter_time,
                _sender_uid=self.real_uid_counter,
                _receiver_uid=None,
                _real_encounter_time=observed_risk.encounter_time,
                _exposition_event=(
                    observed_risk.encounter_time >= exposure_timestamp),
            )

            user_messages.append(encounter_message)
            last_risk = encounter_message.risk_level
            for update_signature in observed_risk.update_signatures:
                assert update_signature.update_risk_level <= mu.message_uid_mask

                update_message = mu.UpdateMessage(
                    uid=encounter_message.uid,
                    old_risk_level=last_risk,
                    new_risk_level=update_signature.update_risk_level,
                    encounter_time=encounter_message.encounter_time,
                    update_time=update_signature.update_time,
                    _sender_uid=encounter_message._sender_uid,
                    _receiver_uid=encounter_message._receiver_uid,
                    _real_encounter_time=encounter_message._real_encounter_time,
                    _real_update_time=update_signature.update_time,
                    _update_reason=update_signature.update_reason,
                )

                user_messages.append(update_message)
                last_risk = update_signature.update_risk_level
        self.real_uid_counter += 1
        return user_messages

    def generate_random_messages(
            self,
            max_timestamp: mu.TimestampType,
            exposure_timestamp: mu.TimestampType,
            n_encounters: int,
            n_messages: int,
            min_risk_level: mu.RiskLevelType = 0,
            max_risk_level: mu.RiskLevelType = mu.risk_level_mask
    ) -> typing.List[mu.GenericMessageType]:
        """
        Returns a set of random encounter/update messages
        """
        assert 1 <= n_encounters <= n_messages
        risk_levels = np.random.randint(
            min_risk_level, max_risk_level + 1,
            size=n_messages).astype(mu.RiskLevelType)
        message_times = np.random.choice(
            max_timestamp, size=n_messages).astype(mu.TimestampType)
        partitions = np.random.choice(np.arange(1, n_messages),
                                      size=n_encounters-1, replace=False)
        partitions.sort()
        risk_levels = np.split(risk_levels, partitions)
        message_times = np.split(message_times, partitions)
        observed_risks = [None] * n_encounters
        for i in range(n_encounters):
            message_times[i].sort()
            n_updates = message_times[i].shape[0] - 1

            encounter_risk = risk_levels[i][0]
            encounter_time = message_times[i][0]

            update_risks = risk_levels[i][1:]
            update_times = message_times[i][1:]

            observed_risks[i] = ObservedRisk(
                encounter_time=encounter_time,
                encounter_risk_level=encounter_risk
            )
            for j in range(n_updates):
                update_time = update_times[j]
                update_risk_level = update_risks[j]
                observed_risks[i].update(
                    update_time=update_time,
                    update_risk_level=update_risk_level
                )
        return self.compile_message_history(observed_risks)

    @staticmethod
    def _get_linear_saturation_observed_risks(
        max_timestamp: mu.TimestampType,
        exposure_timestamp: mu.TimestampType,
        risk_level_low: mu.RiskLevelType,
        risk_level_high: mu.RiskLevelType,
        rate: int = 1
    ) -> typing.List[mu.GenericMessageType]:
        """
        Returns the observed risks belonging to a linear saturation curve (_/¯)
        """
        if not exposure_timestamp:
            exposure_timestamp = max_timestamp + 1
        positive_test_time = exposure_timestamp + \
            (risk_level_high - risk_level_low) * rate

        if positive_test_time <= max_timestamp:

            def saturating_encounter_risk(t):
                if t <= exposure_timestamp:
                    return risk_level_low
                elif t >= positive_test_time:
                    return risk_level_high
                else:
                    return (t - exposure_timestamp) / rate

            def saturating_update_message(t):
                if exposure_timestamp < t < positive_test_time:
                    return [
                        ObservedRisk.UpdateSignature(
                            positive_test_time, risk_level_high, None
                        )
                    ]
                else:
                    return []

            observed_risks = [
                ObservedRisk(
                    encounter_time=t,
                    encounter_risk_level=saturating_encounter_risk(t),
                    update_signatures=saturating_update_message(t)
                )
                for t in range(max_timestamp)
            ]
        else:
            observed_risks = [
                ObservedRisk(
                    encounter_time=t,
                    encounter_risk_level=risk_level_low,
                    update_signatures=[]
                )
                for t in range(max_timestamp)
            ]
        return observed_risks

    def generate_linear_saturation_risk_messages(
            self,
            max_timestamp: mu.TimestampType,
            exposure_timestamp: mu.TimestampType,
            n_encounters: int,
            init_risk_level: mu.RiskLevelType = 0,
            final_risk_level: mu.RiskLevelType = mu.message_uid_mask,
    ) -> typing.List[mu.GenericMessageType]:
        """
        Returns a set of random encounter/update messages sampled from
        observed risks belonging to a linear saturation curve (_/¯)
        """
        assert n_encounters <= max_timestamp, \
            'In linear saturation messages, no two encounters happen \
                in one timestamp'
        # create at least one encounter
        linear_saturation_risks = self._get_linear_saturation_observed_risks(
            max_timestamp=max_timestamp,
            exposure_timestamp=exposure_timestamp,
            risk_level_low=init_risk_level,
            risk_level_high=final_risk_level,
        )

        inds = np.random.choice(
            max_timestamp, size=n_encounters, replace=False)
        sample_risks = [linear_saturation_risks[ind] for ind in inds]
        return self.compile_message_history(sample_risks)
