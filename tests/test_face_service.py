import pytest

from app.services import face_service


def test_cosine_distance_identical_vectors_is_zero():
    vec = [1.0, 2.0, 3.0]
    assert face_service.cosine_distance(vec, vec) == pytest.approx(0.0, abs=1e-9)


def test_cosine_distance_orthogonal_vectors_is_one():
    assert face_service.cosine_distance([1.0, 0.0], [0.0, 1.0]) == pytest.approx(1.0)


def test_average_embeddings_computes_mean():
    result = face_service.average_embeddings([[1.0, 2.0], [3.0, 4.0]])
    assert result == pytest.approx([2.0, 3.0])


def test_average_embeddings_raises_on_empty():
    with pytest.raises(ValueError):
        face_service.average_embeddings([])


def test_check_same_person_embeddings_passes_for_near_identical_vectors():
    # 거의 동일한 벡터 3개 -> 같은 사람으로 간주, 예외 없어야 함
    base = [1.0, 0.0, 0.0, 0.0]
    close = [0.99, 0.01, 0.0, 0.0]
    face_service.check_same_person_embeddings([base, close, base])


def test_check_same_person_embeddings_raises_for_different_person():
    # 서로 완전히 다른(직교) 벡터가 섞이면 다른 인물로 판단해야 함
    person_a = [1.0, 0.0, 0.0, 0.0]
    person_b = [0.0, 1.0, 0.0, 0.0]
    with pytest.raises(face_service.DifferentPersonError):
        face_service.check_same_person_embeddings([person_a, person_a, person_b])


def test_check_same_person_embeddings_single_embedding_is_noop():
    # 셀피 1장뿐이면 비교 대상이 없으므로 그냥 통과
    face_service.check_same_person_embeddings([[1.0, 0.0, 0.0]])


def test_check_same_person_embeddings_empty_is_noop():
    face_service.check_same_person_embeddings([])


def _face(embedding, x=0, y=0, w=100, h=100, confidence=0.99):
    return face_service.DetectedFace(
        embedding=embedding,
        facial_area={"x": x, "y": y, "w": w, "h": h},
        confidence=confidence,
    )


def test_select_matching_face_picks_closest_to_reference():
    # 기준 벡터와 가장 가까운 얼굴 하나만 골라야 한다 (기능명세서 2.2).
    reference = [1.0, 0.0, 0.0, 0.0]
    bystander = _face([0.0, 1.0, 0.0, 0.0], x=0)  # 완전히 다른 사람 (직교 벡터)
    registered_user = _face([0.99, 0.01, 0.0, 0.0], x=200)  # 기준 벡터와 거의 동일

    best = face_service.select_matching_face([bystander, registered_user], reference)

    assert best is registered_user


def test_select_matching_face_raises_when_no_face_is_close_enough():
    # 검출된 얼굴이 있어도, 가장 가까운 것마저 임계값을 넘으면 본인이 아닌 것으로 판단.
    reference = [1.0, 0.0, 0.0, 0.0]
    stranger_a = _face([0.0, 1.0, 0.0, 0.0])
    stranger_b = _face([0.0, 0.0, 1.0, 0.0])

    with pytest.raises(face_service.PersonMismatchError):
        face_service.select_matching_face([stranger_a, stranger_b], reference)


def test_select_matching_face_raises_no_face_detected_when_list_empty():
    with pytest.raises(face_service.NoFaceDetectedError):
        face_service.select_matching_face([], [1.0, 0.0, 0.0, 0.0])


def test_select_matching_face_single_face_within_threshold_passes():
    reference = [1.0, 0.0, 0.0, 0.0]
    only_face = _face([0.98, 0.02, 0.0, 0.0])

    best = face_service.select_matching_face([only_face], reference)

    assert best is only_face
