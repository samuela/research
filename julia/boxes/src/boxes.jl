struct Box
  mins
  maxs
end

function intersect_boxes(b1, b2)
  Box(max.(b1.mins, b2.mins), min.(b1.maxs, b2.maxs))
end

is_empty(box) = any(box.mins .> box.maxs)
in_box(box, x) = all(box.mins .<= x) && all(x .<= box.maxs)
rand_inside(box) = box.mins + (box.maxs - box.mins) .* rand(length(box.mins))

# TODO would performance be better with more type hints? Right now we're getting
# a buncy of Any's.
struct BoxSet
  # Array{Box}
  boxes
  # `adjacency` is a Dict box_id -> [box_id]
  adjacency
end

# !. doesn't work so we alias
not = !
function add_box!(box_set::BoxSet, box)
  # TODO: it may make sense to store the intersection so we don't have to bother
  # with it later.
  # Make sure to calculate neighbors before adding to `box_set.boxes`!
  neighbors = findall(not.(is_empty.(intersect_boxes.(box_set.boxes, Ref(box)))))

  push!(box_set.boxes, box)
  id = length(box_set.boxes)

  box_set.adjacency[id] = neighbors
  for id2 in neighbors
    push!(box_set.adjacency[id2], id)
  end

  id
end
