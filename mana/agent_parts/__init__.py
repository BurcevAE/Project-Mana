"""mana.agent_parts — mixin classes that together compose mana.agent.ManaAgent.

Each file below owns one concern of the agent and can be read, tested and
changed independently. mana/agent.py just does:

    class ManaAgent(CoreMixin, ContextMixin, RoutingMixin, ConfidenceMixin,
                     ExecutionMixin, BenchmarkingMixin, EvolutionMixin,
                     KnowledgeOpsMixin):
        """
