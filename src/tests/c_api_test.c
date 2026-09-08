#include "jaide.h"

#include <math.h>
#include <stdio.h>
#include <string.h>

static int g_failures = 0;
static int g_passes = 0;

static void check(const char *name, int cond) {
    if (cond) {
        g_passes++;
        fprintf(stdout, "[PASS] %s\n", name);
    } else {
        g_failures++;
        fprintf(stderr, "[FAIL] %s\n", name);
    }
}

static int approx(double a, double b, double eps) {
    double d = a - b;
    if (d < 0.0) {
        d = -d;
    }
    return d <= eps;
}

static void test_version(void) {
    check("version major is 4", jaide_version_major() == 4);
    check("version minor is non-negative", jaide_version_minor() >= 0);
    check("version patch is non-negative", jaide_version_patch() >= 0);
}

static void test_error_strings(void) {
    const char *ok = jaide_get_error_string(JAIDE_SUCCESS);
    const char *null_ptr = jaide_get_error_string(JAIDE_ERROR_NULL_POINTER);
    const char *node_missing = jaide_get_error_string(JAIDE_ERROR_NODE_NOT_FOUND);
    check("success string is non-null", ok != NULL && ok[0] != '\0');
    check("null pointer string is non-null", null_ptr != NULL && null_ptr[0] != '\0');
    check("node not found string is non-null", node_missing != NULL && node_missing[0] != '\0');
    check("distinct codes map to distinct strings", strcmp(ok, null_ptr) != 0);
}

static void test_null_handling(void) {
    check("add node on null graph", jaide_add_node(NULL, "a", "t") == JAIDE_ERROR_NULL_POINTER);
    check("add edge on null graph", jaide_add_edge(NULL, "a", "b", 0.5) == JAIDE_ERROR_NULL_POINTER);
    check("clear on null graph", jaide_clear_graph(NULL) == JAIDE_ERROR_NULL_POINTER);
    check("node count on null graph", jaide_graph_node_count(NULL) == 0);
    check("edge count on null graph", jaide_graph_edge_count(NULL) == 0);
    check("destroy null graph is safe", (jaide_destroy_graph(NULL), 1));
    check("destroy null optimizer is safe", (jaide_destroy_optimizer(NULL), 1));
}

static void test_node_lifecycle(JaideGraph *g) {
    check("graph starts empty", jaide_graph_node_count(g) == 0);
    check("add node alpha", jaide_add_node(g, "alpha", "concept") == JAIDE_SUCCESS);
    check("add node beta", jaide_add_node(g, "beta", "concept") == JAIDE_SUCCESS);
    check("duplicate node rejected", jaide_add_node(g, "alpha", "concept") == JAIDE_ERROR_DUPLICATE_NODE);
    check("empty id rejected", jaide_add_node(g, "", "concept") == JAIDE_ERROR_INVALID_STRING);
    check("node count is two", jaide_graph_node_count(g) == 2);
    check("has node alpha", jaide_has_node(g, "alpha") == 1);
    check("has node gamma is false", jaide_has_node(g, "gamma") == 0);

    check("add node gamma", jaide_add_node(g, "gamma", "concept") == JAIDE_SUCCESS);
    check("node count is three", jaide_graph_node_count(g) == 3);
    check("remove node gamma", jaide_remove_node(g, "gamma") == JAIDE_SUCCESS);
    check("node count back to two", jaide_graph_node_count(g) == 2);
    check("remove missing node", jaide_remove_node(g, "gamma") == JAIDE_ERROR_NODE_NOT_FOUND);
}

static void test_quantum_state(JaideGraph *g) {
    JaideQuantumState state;
    memset(&state, 0, sizeof(state));

    check("read default state", jaide_get_node_quantum_state(g, "alpha", &state) == JAIDE_SUCCESS);
    check("default amplitude is one", approx(state.real, 1.0, 1e-12) && approx(state.imag, 0.0, 1e-12));

    check("set state", jaide_set_node_quantum_state(g, "alpha", 3.0, 4.0) == JAIDE_SUCCESS);
    check("read set state", jaide_get_node_quantum_state(g, "alpha", &state) == JAIDE_SUCCESS);
    check("state is normalized real", approx(state.real, 0.6, 1e-12));
    check("state is normalized imag", approx(state.imag, 0.8, 1e-12));
    check("probability is unit", approx(jaide_get_node_probability(g, "alpha"), 1.0, 1e-9));
    check("magnitude is unit", approx(jaide_get_node_magnitude(g, "alpha"), 1.0, 1e-9));

    check("non finite state rejected", jaide_set_node_quantum_state(g, "alpha", INFINITY, 0.0) == JAIDE_ERROR_INVALID_PARAMETER);
    check("state on missing node", jaide_set_node_quantum_state(g, "missing", 1.0, 0.0) == JAIDE_ERROR_NODE_NOT_FOUND);
    check("null out state rejected", jaide_get_node_quantum_state(g, "alpha", NULL) == JAIDE_ERROR_NULL_POINTER);

    check("set phase", jaide_set_node_phase(g, "alpha", 1.25) == JAIDE_SUCCESS);
    check("get phase", approx(jaide_get_node_phase(g, "alpha"), 1.25, 1e-12));
}

static void test_gates(JaideGraph *g) {
    JaideQuantumState before;
    JaideQuantumState after;
    const double inv_sqrt2 = 0.7071067811865476;

    check("reset state for identity", jaide_set_node_quantum_state(g, "beta", 1.0, 0.0) == JAIDE_SUCCESS);
    check("read pre-gate state", jaide_get_node_quantum_state(g, "beta", &before) == JAIDE_SUCCESS);
    check("identity gate", jaide_apply_identity_gate(g, "beta") == JAIDE_SUCCESS);
    check("read post-identity state", jaide_get_node_quantum_state(g, "beta", &after) == JAIDE_SUCCESS);
    check("identity preserves amplitude", approx(before.real, after.real, 1e-12) && approx(before.imag, after.imag, 1e-12));
    check("identity keeps probability one", approx(jaide_get_node_probability(g, "beta"), 1.0, 1e-12));
    check("identity keeps unit magnitude", approx(jaide_get_node_magnitude(g, "beta"), 1.0, 1e-12));

    check("reset state for hadamard", jaide_set_node_quantum_state(g, "beta", 1.0, 0.0) == JAIDE_SUCCESS);
    check("hadamard gate", jaide_apply_hadamard(g, "beta") == JAIDE_SUCCESS);
    check("read post-hadamard state", jaide_get_node_quantum_state(g, "beta", &after) == JAIDE_SUCCESS);
    check("hadamard splits amplitude", approx(after.real, inv_sqrt2, 1e-12) && approx(after.imag, 0.0, 1e-12));
    check("hadamard halves probability", approx(jaide_get_node_probability(g, "beta"), 0.5, 1e-12));
    check("hadamard keeps unit magnitude", approx(jaide_get_node_magnitude(g, "beta"), 1.0, 1e-12));

    check("reset state for pauli x", jaide_set_node_quantum_state(g, "beta", 1.0, 0.0) == JAIDE_SUCCESS);
    check("pauli x gate", jaide_apply_pauli_x(g, "beta") == JAIDE_SUCCESS);
    check("read post-pauli-x state", jaide_get_node_quantum_state(g, "beta", &after) == JAIDE_SUCCESS);
    check("pauli x empties first amplitude", approx(after.real, 0.0, 1e-12) && approx(after.imag, 0.0, 1e-12));
    check("pauli x zeroes probability", approx(jaide_get_node_probability(g, "beta"), 0.0, 1e-12));
    check("pauli x keeps unit magnitude", approx(jaide_get_node_magnitude(g, "beta"), 1.0, 1e-12));

    check("reset state for pauli y", jaide_set_node_quantum_state(g, "beta", 1.0, 0.0) == JAIDE_SUCCESS);
    check("pauli y gate", jaide_apply_pauli_y(g, "beta") == JAIDE_SUCCESS);
    check("read post-pauli-y state", jaide_get_node_quantum_state(g, "beta", &after) == JAIDE_SUCCESS);
    check("pauli y empties first amplitude", approx(after.real, 0.0, 1e-12) && approx(after.imag, 0.0, 1e-12));
    check("pauli y zeroes probability", approx(jaide_get_node_probability(g, "beta"), 0.0, 1e-12));
    check("pauli y keeps unit magnitude", approx(jaide_get_node_magnitude(g, "beta"), 1.0, 1e-12));

    check("reset state for pauli z", jaide_set_node_quantum_state(g, "beta", 1.0, 0.0) == JAIDE_SUCCESS);
    check("pauli z gate", jaide_apply_pauli_z(g, "beta") == JAIDE_SUCCESS);
    check("read post-pauli-z state", jaide_get_node_quantum_state(g, "beta", &after) == JAIDE_SUCCESS);
    check("pauli z preserves first amplitude", approx(after.real, 1.0, 1e-12) && approx(after.imag, 0.0, 1e-12));
    check("pauli z keeps probability one", approx(jaide_get_node_probability(g, "beta"), 1.0, 1e-12));
    check("pauli z keeps unit magnitude", approx(jaide_get_node_magnitude(g, "beta"), 1.0, 1e-12));

    check("reset state for double hadamard", jaide_set_node_quantum_state(g, "beta", 1.0, 0.0) == JAIDE_SUCCESS);
    check("gate by numeric id", jaide_apply_gate(g, "beta", JAIDE_GATE_HADAMARD) == JAIDE_SUCCESS);
    check("gate by numeric id twice", jaide_apply_gate(g, "beta", JAIDE_GATE_HADAMARD) == JAIDE_SUCCESS);
    check("read post-double-hadamard state", jaide_get_node_quantum_state(g, "beta", &after) == JAIDE_SUCCESS);
    check("double hadamard is identity", approx(after.real, 1.0, 1e-12) && approx(after.imag, 0.0, 1e-12));

    check("unknown gate rejected", jaide_apply_gate(g, "beta", 99) == JAIDE_ERROR_UNKNOWN_GATE);
    check("gate on missing node", jaide_apply_gate(g, "missing", JAIDE_GATE_HADAMARD) == JAIDE_ERROR_NODE_NOT_FOUND);
}

static void test_edges(JaideGraph *g) {
    check("edge count starts at zero", jaide_graph_edge_count(g) == 0);
    check("add edge", jaide_add_edge(g, "alpha", "beta", 0.75) == JAIDE_SUCCESS);
    check("edge count is one", jaide_graph_edge_count(g) == 1);
    check("has edge", jaide_has_edge(g, "alpha", "beta") == 1);
    check("reverse edge absent", jaide_has_edge(g, "beta", "alpha") == 0);
    check("edge weight round trip", approx(jaide_get_edge_weight(g, "alpha", "beta"), 0.75, 1e-12));

    check("self edge rejected", jaide_add_edge(g, "alpha", "alpha", 0.5) == JAIDE_ERROR_SELF_REFERENCE);
    check("edge to missing node", jaide_add_edge(g, "alpha", "missing", 0.5) == JAIDE_ERROR_NODE_NOT_FOUND);

    check("weight clamped high", jaide_set_edge_weight(g, "alpha", "beta", 5.0) == JAIDE_SUCCESS);
    check("clamped weight is one", approx(jaide_get_edge_weight(g, "alpha", "beta"), 1.0, 1e-12));
    check("weight clamped low", jaide_set_edge_weight(g, "alpha", "beta", -5.0) == JAIDE_SUCCESS);
    check("clamped weight is zero", approx(jaide_get_edge_weight(g, "alpha", "beta"), 0.0, 1e-12));
    check("set weight on missing edge", jaide_set_edge_weight(g, "beta", "alpha", 0.5) == JAIDE_ERROR_EDGE_NOT_FOUND);

    check("default quality is coherent", jaide_get_edge_quality(g, "alpha", "beta") == 2);
    check("set quality entangled", jaide_set_edge_quality(g, "alpha", "beta", 1) == JAIDE_SUCCESS);
    check("quality round trip", jaide_get_edge_quality(g, "alpha", "beta") == 1);
    check("invalid quality rejected", jaide_set_edge_quality(g, "alpha", "beta", 42) == JAIDE_ERROR_INVALID_QUALITY);

    check("set fractal dimension", jaide_set_edge_fractal_dimension(g, "alpha", "beta", 1.5) == JAIDE_SUCCESS);
    check("fractal dimension round trip", approx(jaide_get_edge_fractal_dimension(g, "alpha", "beta"), 1.5, 1e-12));
    check("correlation magnitude non-negative", jaide_get_edge_correlation_magnitude(g, "alpha", "beta") >= 0.0);

    check("entangle nodes", jaide_entangle_nodes(g, "alpha", "beta") == JAIDE_SUCCESS);
    check("remove edge", jaide_remove_edge(g, "alpha", "beta") == JAIDE_SUCCESS);
    check("remove missing edge", jaide_remove_edge(g, "alpha", "beta") == JAIDE_ERROR_EDGE_NOT_FOUND);
}

static void test_cascade_removal(void) {
    JaideGraph *g = jaide_create_graph();
    check("cascade graph created", g != NULL);
    if (g == NULL) return;

    check("cascade graph starts empty", jaide_graph_node_count(g) == 0 && jaide_graph_edge_count(g) == 0);
    check("add cascade source", jaide_add_node(g, "casc_a", "concept") == JAIDE_SUCCESS);
    check("add cascade target", jaide_add_node(g, "casc_b", "concept") == JAIDE_SUCCESS);
    check("add cascade spare", jaide_add_node(g, "casc_c", "concept") == JAIDE_SUCCESS);
    check("add cascade edge ab", jaide_add_edge(g, "casc_a", "casc_b", 0.5) == JAIDE_SUCCESS);
    check("add cascade edge ac", jaide_add_edge(g, "casc_a", "casc_c", 0.5) == JAIDE_SUCCESS);
    check("add cascade edge bc", jaide_add_edge(g, "casc_b", "casc_c", 0.5) == JAIDE_SUCCESS);
    check("cascade edges present", jaide_graph_edge_count(g) == 3);
    check("remove cascade source", jaide_remove_node(g, "casc_a") == JAIDE_SUCCESS);
    check("cascade incident edges removed", jaide_graph_edge_count(g) == 1);
    check("cascade node removed", jaide_has_node(g, "casc_a") == 0);
    check("cascade target survives", jaide_has_node(g, "casc_b") == 1);
    check("cascade spare survives", jaide_has_node(g, "casc_c") == 1);
    check("cascade node count", jaide_graph_node_count(g) == 2);
    check("remove cascade target", jaide_remove_node(g, "casc_b") == JAIDE_SUCCESS);
    check("cascade last edge removed", jaide_graph_edge_count(g) == 0);
    check("remove cascade spare", jaide_remove_node(g, "casc_c") == JAIDE_SUCCESS);
    check("cascade graph ends empty", jaide_graph_node_count(g) == 0);

    jaide_destroy_graph(g);
}

static void test_information_codec(JaideGraph *g) {
    char node_id[256];
    unsigned char decoded[256];
    const char *payload = "jaide-inference-trace";

    memset(node_id, 0, sizeof(node_id));
    memset(decoded, 0, sizeof(decoded));

    check("encode information", jaide_encode_information(g, (const unsigned char *)payload, node_id, sizeof(node_id)) == JAIDE_SUCCESS);
    check("encoded node id is non-empty", node_id[0] != '\0');
    check("encoded node exists", jaide_has_node(g, node_id) == 1);
    check("decode information", jaide_decode_information(g, node_id, decoded, sizeof(decoded)) == JAIDE_SUCCESS);
    check("decoded payload matches", strcmp((const char *)decoded, payload) == 0);

    check("encode with zero buffer", jaide_encode_information(g, (const unsigned char *)payload, node_id, 0) == JAIDE_ERROR_INVALID_PARAMETER);
    check("decode unknown node", jaide_decode_information(g, "no_such_node", decoded, sizeof(decoded)) == JAIDE_ERROR_NODE_NOT_FOUND);
    check("remove encoded node", jaide_remove_node(g, node_id) == JAIDE_SUCCESS);
}

static void test_node_data(JaideGraph *g) {
    unsigned char data[64];
    memset(data, 0xFF, sizeof(data));
    check("add typed node", jaide_add_node(g, "typed", "relational-payload") == JAIDE_SUCCESS);
    check("read node data", jaide_get_node_data(g, "typed", data, sizeof(data)) == JAIDE_SUCCESS);
    check("node data matches type", strcmp((const char *)data, "relational-payload") == 0);
    check("node data zero buffer", jaide_get_node_data(g, "typed", data, 0) == JAIDE_ERROR_INVALID_PARAMETER);
    check("node data missing node", jaide_get_node_data(g, "absent", data, sizeof(data)) == JAIDE_ERROR_NODE_NOT_FOUND);
    check("remove typed node", jaide_remove_node(g, "typed") == JAIDE_SUCCESS);
}

static void test_topology_hash(JaideGraph *g) {
    unsigned char hash_a[128];
    unsigned char hash_b[128];
    unsigned char hash_c[128];

    memset(hash_a, 0, sizeof(hash_a));
    memset(hash_b, 0, sizeof(hash_b));
    memset(hash_c, 0, sizeof(hash_c));

    check("topology hash first read", jaide_get_topology_hash(g, hash_a, sizeof(hash_a)) == JAIDE_SUCCESS);
    check("topology hash is non-empty", hash_a[0] != 0);
    check("topology hash second read", jaide_get_topology_hash(g, hash_b, sizeof(hash_b)) == JAIDE_SUCCESS);
    check("topology hash is deterministic", memcmp(hash_a, hash_b, sizeof(hash_a)) == 0);

    check("add hash-affecting node", jaide_add_node(g, "hash_probe", "concept") == JAIDE_SUCCESS);
    check("topology hash third read", jaide_get_topology_hash(g, hash_c, sizeof(hash_c)) == JAIDE_SUCCESS);
    check("topology hash tracks structure", memcmp(hash_a, hash_c, sizeof(hash_a)) != 0);
    check("remove hash-affecting node", jaide_remove_node(g, "hash_probe") == JAIDE_SUCCESS);
    check("topology hash zero buffer", jaide_get_topology_hash(g, hash_a, 0) == JAIDE_ERROR_INVALID_PARAMETER);
}

static void test_fractal_dimension(JaideGraph *g) {
    double dimension = jaide_get_fractal_dimension(g);
    check("fractal dimension is finite", isfinite(dimension));
    check("fractal dimension is non-negative", dimension >= 0.0);
}

static void test_optimizer(void) {
    JaideGraph *g = jaide_create_graph();
    JaideOptimizer *opt = NULL;
    int iterations = -1;
    double best_energy = 0.0;
    double acceptance_rate = -1.0;

    check("optimizer graph created", g != NULL);
    if (g == NULL) {
        return;
    }

    check("optimizer node one", jaide_add_node(g, "n1", "concept") == JAIDE_SUCCESS);
    check("optimizer node two", jaide_add_node(g, "n2", "concept") == JAIDE_SUCCESS);
    check("optimizer node three", jaide_add_node(g, "n3", "concept") == JAIDE_SUCCESS);
    check("optimizer edge one", jaide_add_edge(g, "n1", "n2", 0.4) == JAIDE_SUCCESS);
    check("optimizer edge two", jaide_add_edge(g, "n2", "n3", 0.6) == JAIDE_SUCCESS);

    opt = jaide_create_optimizer(1.0, 0.9, 32);
    check("optimizer created", opt != NULL);

    if (opt != NULL) {
        check("optimizer cooling rate accepted", jaide_set_optimizer_config(opt, "cooling_rate", 0.95) == JAIDE_SUCCESS);
        check("optimizer min temperature accepted", jaide_set_optimizer_config(opt, "min_temperature", 1e-4) == JAIDE_SUCCESS);
        check("optimizer edge probability accepted", jaide_set_optimizer_config(opt, "perturb_edge_prob", 0.75) == JAIDE_SUCCESS);
        check("optimizer node probability accepted", jaide_set_optimizer_config(opt, "perturb_node_prob", 0.25) == JAIDE_SUCCESS);
        check("optimizer reheat factor accepted", jaide_set_optimizer_config(opt, "reheat_factor", 1.5) == JAIDE_SUCCESS);
        check("optimizer max iterations accepted", jaide_set_optimizer_config(opt, "max_iterations", 64.0) == JAIDE_SUCCESS);
        check("optimizer adaptive cooling accepted", jaide_set_optimizer_config(opt, "adaptive_cooling", 0.0) == JAIDE_SUCCESS);
        check("optimizer unknown key rejected", jaide_set_optimizer_config(opt, "not_a_key", 1.0) == JAIDE_ERROR_INVALID_PARAMETER);
        check("optimizer non-finite value rejected", jaide_set_optimizer_config(opt, "cooling_rate", INFINITY) == JAIDE_ERROR_INVALID_PARAMETER);
        check("optimizer out-of-range cooling rejected", jaide_set_optimizer_config(opt, "cooling_rate", 1.0) == JAIDE_ERROR_INVALID_PARAMETER);
        check("optimizer out-of-range probability rejected", jaide_set_optimizer_config(opt, "perturb_edge_prob", 1.5) == JAIDE_ERROR_INVALID_PARAMETER);
        check("optimizer null key rejected", jaide_set_optimizer_config(opt, NULL, 0.5) == JAIDE_ERROR_NULL_POINTER);
        check("optimizer null handle rejected", jaide_set_optimizer_config(NULL, "cooling_rate", 0.5) == JAIDE_ERROR_NULL_POINTER);
        check("optimizer run", jaide_optimize_graph(opt, g) == JAIDE_SUCCESS);
        check("optimizer statistics", jaide_get_optimizer_statistics(opt, &iterations, &best_energy, &acceptance_rate) == JAIDE_SUCCESS);
        check("optimizer iterations recorded", iterations > 0);
        check("optimizer iterations bounded", iterations <= 64);
        check("optimizer energy finite", isfinite(best_energy));
        check("optimizer acceptance in range", acceptance_rate >= 0.0 && acceptance_rate <= 1.0);
        check("optimizer statistics tolerate null outputs", jaide_get_optimizer_statistics(opt, NULL, NULL, NULL) == JAIDE_SUCCESS);
        check("optimizer null graph rejected", jaide_optimize_graph(opt, NULL) == JAIDE_ERROR_NULL_POINTER);
        check("optimizer graph intact", jaide_graph_node_count(g) == 3);
        check("optimizer edges intact", jaide_graph_edge_count(g) == 2);
        jaide_destroy_optimizer(opt);
    }

    check("optimizer null run rejected", jaide_optimize_graph(NULL, g) == JAIDE_ERROR_NULL_POINTER);
    jaide_destroy_graph(g);
}

static void test_clear(JaideGraph *g) {
    check("clear graph", jaide_clear_graph(g) == JAIDE_SUCCESS);
    check("nodes cleared", jaide_graph_node_count(g) == 0);
    check("edges cleared", jaide_graph_edge_count(g) == 0);
}

int main(int argc, char **argv) {
    JaideGraph *graph = NULL;

    (void)argc;
    (void)argv;

    setvbuf(stdout, NULL, _IONBF, 0);
    fprintf(stdout, "jaide-c-api-test starting\n");

    test_version();
    test_error_strings();
    test_null_handling();

    graph = jaide_create_graph();
    check("graph created", graph != NULL);
    if (graph == NULL) {
        fprintf(stdout, "jaide-c-api-test: %d passed, %d failed\n", g_passes, g_failures);
        return 1;
    }

    test_node_lifecycle(graph);
    test_quantum_state(graph);
    test_gates(graph);
    test_edges(graph);
    test_cascade_removal();
    test_information_codec(graph);
    test_node_data(graph);
    test_topology_hash(graph);
    test_fractal_dimension(graph);
    test_clear(graph);

    jaide_destroy_graph(graph);

    test_optimizer();

    fprintf(stdout, "jaide-c-api-test: %d passed, %d failed\n", g_passes, g_failures);
    return g_failures == 0 ? 0 : 1;
}
