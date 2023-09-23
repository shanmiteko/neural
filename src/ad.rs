use std::cell::{Ref, RefCell, RefMut};
use std::fmt::Debug;
use std::ops::{Add, Div, Mul, Sub};
use std::rc::Rc;

type Child = (f64, PtrVar);
type Childrn = Vec<Child>;

pub struct Var {
    value: f64,
    grad: f64,
    children: Childrn,
}

impl Debug for Var {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.value)
    }
}

impl Default for Var {
    fn default() -> Self {
        Self {
            value: 0.,
            grad: 0.,
            children: vec![],
        }
    }
}

impl From<f64> for Var {
    fn from(value: f64) -> Self {
        Self {
            value,
            ..Default::default()
        }
    }
}

impl Var {
    fn calc_grad(&mut self, grad: Option<f64>) {
        let grad = grad.unwrap_or(1.);
        self.grad += grad;
        for (coef, child) in &self.children {
            child.borrow_mut().calc_grad(Some(grad * coef));
        }
    }
}

#[derive(Clone)]
pub struct PtrVar(Rc<RefCell<Var>>);

impl Debug for PtrVar {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.value())
    }
}

impl From<f64> for PtrVar {
    fn from(value: f64) -> Self {
        Self::new(value)
    }
}

impl PtrVar {
    pub fn new<V: Into<Var>>(var: V) -> Self {
        Self(Rc::new(RefCell::new(var.into())))
    }

    fn borrow(&self) -> Ref<'_, Var> {
        self.0.borrow()
    }

    fn borrow_mut(&self) -> RefMut<'_, Var> {
        self.0.borrow_mut()
    }

    pub fn calc_grad(&self) {
        self.borrow_mut().calc_grad(None)
    }

    pub fn value(&self) -> f64 {
        self.borrow().value
    }

    pub fn value_mut(&self, value: f64) {
        self.borrow_mut().value = value
    }

    pub fn grad(&self) -> f64 {
        self.borrow().grad
    }

    pub fn grad_mut(&self, grad: f64) {
        self.borrow_mut().grad = grad
    }
}

impl Add for PtrVar {
    type Output = PtrVar;

    fn add(self, rhs: Self) -> Self::Output {
        PtrVar::new(Var {
            value: self.0.borrow().value + rhs.0.borrow().value,
            children: vec![(1., self.clone()), (1., rhs.clone())],
            ..Default::default()
        })
    }
}

impl Sub for PtrVar {
    type Output = PtrVar;

    fn sub(self, rhs: Self) -> Self::Output {
        PtrVar::new(Var {
            value: self.borrow().value - rhs.borrow().value,
            children: vec![(1., self.clone()), (-1., rhs.clone())],
            ..Default::default()
        })
    }
}

impl Mul for PtrVar {
    type Output = PtrVar;

    fn mul(self, rhs: Self) -> Self::Output {
        PtrVar::new(Var {
            value: self.borrow().value * rhs.borrow().value,
            children: vec![
                (rhs.borrow().value, self.clone()),
                (self.borrow().value, rhs.clone()),
            ],
            ..Default::default()
        })
    }
}

impl Div for PtrVar {
    type Output = PtrVar;

    fn div(self, rhs: Self) -> Self::Output {
        PtrVar::new(Var {
            value: self.borrow().value / rhs.borrow().value,
            children: vec![
                (1. / rhs.borrow().value, self.clone()),
                (
                    self.borrow().value * (-(1. / rhs.borrow().value.powi(2))),
                    rhs.clone(),
                ),
            ],
            ..Default::default()
        })
    }
}

impl PtrVar {
    pub fn pow(self, rhs: Self) -> Self {
        PtrVar::new(Var {
            value: self.borrow().value.powf(rhs.borrow().value),
            children: vec![
                (
                    rhs.borrow().value * self.borrow().value.powf(rhs.borrow().value - 1.),
                    self.clone(),
                ),
                (
                    self.borrow().value.powf(rhs.borrow().value) * self.borrow().value.ln(),
                    rhs.clone(),
                ),
            ],
            ..Default::default()
        })
    }
}

#[cfg(test)]
mod tests {

    use super::PtrVar;

    #[test]
    fn var_calc_grad() {
        let x = PtrVar::new(2.);
        let y = PtrVar::new(1.);
        // `1/(x**2+y**2)-x*y`
        let f = PtrVar::new(1.) / (x.clone().pow(PtrVar::new(2.)) + y.clone().pow(PtrVar::new(2.)))
            - x.clone() * y.clone();
        f.calc_grad();
        assert_eq!(x.grad(), -1.16);
        assert_eq!(y.grad(), -2.08);
        assert_eq!(f.value(), -1.8);
    }
}
