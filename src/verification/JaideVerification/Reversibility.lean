import JaideVerification.Scalar

set_option autoImplicit false

universe u

namespace Jaide

variable {α : Type u} [RSFField α]

structure LayerParams (α : Type u) [RSFField α] where
  scaleFn : α → α
  transFn : α → α
  scaleFn_ne_zero : ∀ z : α, scaleFn z ≠ RSFField.zero

def couplingForward (p : LayerParams α) (x : α × α) : α × α :=
  (x.1 * p.scaleFn x.2, x.2 + p.transFn (x.1 * p.scaleFn x.2))

def couplingInverse (p : LayerParams α) (y : α × α) : α × α :=
  (y.1 / p.scaleFn (y.2 - p.transFn y.1), y.2 - p.transFn y.1)

def butterflyForward (y : α × α) : α × α :=
  ((y.1 - y.2) * RSFField.scale, (y.1 + y.2) * RSFField.scale)

def butterflyInverse (z : α × α) : α × α :=
  ((z.1 + z.2) * RSFField.scale, (z.2 - z.1) * RSFField.scale)

def layerForward (p : LayerParams α) (x : α × α) : α × α :=
  butterflyForward (couplingForward p x)

def layerInverse (p : LayerParams α) (y : α × α) : α × α :=
  couplingInverse p (butterflyInverse y)

def vectorForward (p : LayerParams α) (v : List (α × α)) : List (α × α) :=
  v.map (layerForward p)

def vectorInverse (p : LayerParams α) (v : List (α × α)) : List (α × α) :=
  v.map (layerInverse p)

def stackForward : List (LayerParams α) → List (α × α) → List (α × α)
  | [], v => v
  | p :: rest, v => stackForward rest (vectorForward p v)

def stackInverse : List (LayerParams α) → List (α × α) → List (α × α)
  | [], v => v
  | p :: rest, v => vectorInverse p (stackInverse rest v)

theorem prodEq {β : Type u} {γ : Type u} {x y : β × γ}
    (h1 : x.1 = y.1) (h2 : x.2 = y.2) : x = y :=
  Eq.trans
    (congrArg (fun c => (x.1, c)) h2)
    (congrArg (fun c => (c, y.2)) h1)

theorem coupling_left_inverse (p : LayerParams α) (x : α × α) :
    couplingInverse p (couplingForward p x) = x :=
  match x with
  | (x1, x2) =>
      have hsnd : x2 + p.transFn (x1 * p.scaleFn x2) - p.transFn (x1 * p.scaleFn x2) = x2 :=
        RSFField.add_sub_cancel x2 (p.transFn (x1 * p.scaleFn x2))
      have hfst :
          x1 * p.scaleFn x2 /
              p.scaleFn (x2 + p.transFn (x1 * p.scaleFn x2) - p.transFn (x1 * p.scaleFn x2)) = x1 :=
        Eq.trans
          (congrArg (fun z => x1 * p.scaleFn x2 / p.scaleFn z) hsnd)
          (RSFField.mul_div_cancel x1 (p.scaleFn x2) (p.scaleFn_ne_zero x2))
      prodEq hfst hsnd

theorem coupling_right_inverse (p : LayerParams α) (y : α × α) :
    couplingForward p (couplingInverse p y) = y :=
  match y with
  | (y1, y2) =>
      have hdiv :
          y1 / p.scaleFn (y2 - p.transFn y1) * p.scaleFn (y2 - p.transFn y1) = y1 :=
        RSFField.div_mul_cancel y1 (p.scaleFn (y2 - p.transFn y1))
          (p.scaleFn_ne_zero (y2 - p.transFn y1))
      have hsnd :
          y2 - p.transFn y1 +
              p.transFn (y1 / p.scaleFn (y2 - p.transFn y1) * p.scaleFn (y2 - p.transFn y1)) = y2 :=
        Eq.trans
          (congrArg (fun z => y2 - p.transFn y1 + p.transFn z) hdiv)
          (RSFField.sub_add_cancel y2 (p.transFn y1))
      prodEq hdiv hsnd

theorem butterfly_left_inverse (y : α × α) :
    butterflyInverse (butterflyForward y) = y :=
  match y with
  | (a, b) =>
      have hfst :
          ((a - b) * RSFField.scale + (a + b) * RSFField.scale) * RSFField.scale = a :=
        Eq.trans
          (congrArg (fun q => q * RSFField.scale)
            (Eq.trans
              (Eq.symm (RSFField.add_mul (a - b) (a + b) RSFField.scale))
              (congrArg (fun q => q * RSFField.scale) (RSFField.sub_add_self a b))))
          (Eq.trans
            (RSFField.mul_assoc (a + a) RSFField.scale RSFField.scale)
            (Eq.trans
              (congrArg (fun q => (a + a) * q) RSFField.scale_sq)
              (RSFField.double_half a)))
      have hsnd :
          ((a + b) * RSFField.scale - (a - b) * RSFField.scale) * RSFField.scale = b :=
        Eq.trans
          (congrArg (fun q => q * RSFField.scale)
            (Eq.trans
              (Eq.symm (RSFField.sub_mul (a + b) (a - b) RSFField.scale))
              (congrArg (fun q => q * RSFField.scale) (RSFField.add_sub_self a b))))
          (Eq.trans
            (RSFField.mul_assoc (b + b) RSFField.scale RSFField.scale)
            (Eq.trans
              (congrArg (fun q => (b + b) * q) RSFField.scale_sq)
              (RSFField.double_half b)))
      prodEq hfst hsnd

theorem butterfly_right_inverse (z : α × α) :
    butterflyForward (butterflyInverse z) = z :=
  match z with
  | (a, b) =>
      have hfst :
          ((a + b) * RSFField.scale - (b - a) * RSFField.scale) * RSFField.scale = a :=
        Eq.trans
          (congrArg (fun q => q * RSFField.scale)
            (Eq.trans
              (Eq.symm (RSFField.sub_mul (a + b) (b - a) RSFField.scale))
              (congrArg (fun q => q * RSFField.scale) (RSFField.add_sub_swap a b))))
          (Eq.trans
            (RSFField.mul_assoc (a + a) RSFField.scale RSFField.scale)
            (Eq.trans
              (congrArg (fun q => (a + a) * q) RSFField.scale_sq)
              (RSFField.double_half a)))
      have hsnd :
          ((a + b) * RSFField.scale + (b - a) * RSFField.scale) * RSFField.scale = b :=
        Eq.trans
          (congrArg (fun q => q * RSFField.scale)
            (Eq.trans
              (Eq.symm (RSFField.add_mul (a + b) (b - a) RSFField.scale))
              (congrArg (fun q => q * RSFField.scale) (RSFField.add_add_self a b))))
          (Eq.trans
            (RSFField.mul_assoc (b + b) RSFField.scale RSFField.scale)
            (Eq.trans
              (congrArg (fun q => (b + b) * q) RSFField.scale_sq)
              (RSFField.double_half b)))
      prodEq hfst hsnd

theorem layer_left_inverse (p : LayerParams α) (x : α × α) :
    layerInverse p (layerForward p x) = x :=
  Eq.trans
    (congrArg (couplingInverse p) (butterfly_left_inverse (couplingForward p x)))
    (coupling_left_inverse p x)

theorem layer_right_inverse (p : LayerParams α) (y : α × α) :
    layerForward p (layerInverse p y) = y :=
  Eq.trans
    (congrArg butterflyForward (coupling_right_inverse p (butterflyInverse y)))
    (butterfly_right_inverse y)

theorem map_left_inverse {β : Type u} (f g : β → β) (h : ∀ x : β, g (f x) = x) :
    ∀ l : List β, (l.map f).map g = l
  | [] => Eq.refl ([] : List β)
  | a :: t =>
      Eq.trans
        (congrArg (fun u => g (f a) :: u) (map_left_inverse f g h t))
        (congrArg (fun u => u :: t) (h a))

theorem vector_left_inverse (p : LayerParams α) (v : List (α × α)) :
    vectorInverse p (vectorForward p v) = v :=
  map_left_inverse (layerForward p) (layerInverse p) (layer_left_inverse p) v

theorem vector_right_inverse (p : LayerParams α) (v : List (α × α)) :
    vectorForward p (vectorInverse p v) = v :=
  map_left_inverse (layerInverse p) (layerForward p) (layer_right_inverse p) v

theorem stack_left_inverse :
    ∀ (ps : List (LayerParams α)) (v : List (α × α)),
      stackInverse ps (stackForward ps v) = v
  | [], v => Eq.refl v
  | p :: rest, v =>
      Eq.trans
        (congrArg (vectorInverse p) (stack_left_inverse rest (vectorForward p v)))
        (vector_left_inverse p v)

theorem stack_right_inverse :
    ∀ (ps : List (LayerParams α)) (v : List (α × α)),
      stackForward ps (stackInverse ps v) = v
  | [], v => Eq.refl v
  | p :: rest, v =>
      Eq.trans
        (congrArg (stackForward rest) (vector_right_inverse p (stackInverse rest v)))
        (stack_right_inverse rest v)

theorem stackForward_injective (ps : List (LayerParams α)) (u v : List (α × α))
    (h : stackForward ps u = stackForward ps v) : u = v :=
  Eq.trans (Eq.symm (stack_left_inverse ps u))
    (Eq.trans (congrArg (stackInverse ps) h) (stack_left_inverse ps v))

theorem stackInverse_injective (ps : List (LayerParams α)) (u v : List (α × α))
    (h : stackInverse ps u = stackInverse ps v) : u = v :=
  Eq.trans (Eq.symm (stack_right_inverse ps u))
    (Eq.trans (congrArg (stackForward ps) h) (stack_right_inverse ps v))

end Jaide
