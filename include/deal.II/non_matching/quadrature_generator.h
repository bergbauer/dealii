// ---------------------------------------------------------------------
//
// Copyright (C) 2021 - 2021 by the deal.II authors
//
// This file is part of the deal.II library.
//
// The deal.II library is free software; you can use it, redistribute
// it, and/or modify it under the terms of the GNU Lesser General
// Public License as published by the Free Software Foundation; either
// version 2.1 of the License, or (at your option) any later version.
// The full text of the license can be found in the file LICENSE.md at
// the top level directory of deal.II.
//
// ---------------------------------------------------------------------

#ifndef dealii_non_matching_quadrature_generator_h
#define dealii_non_matching_quadrature_generator_h

#include <deal.II/base/config.h>

#include <deal.II/base/bounding_box.h>
#include <deal.II/base/function.h>
#include <deal.II/base/quadrature.h>
#include <deal.II/base/smartpointer.h>
#include <deal.II/base/std_cxx17/optional.h>

#include <deal.II/hp/q_collection.h>

#include <deal.II/non_matching/immersed_surface_quadrature.h>

#include <memory>

DEAL_II_NAMESPACE_OPEN
namespace NonMatching
{
  namespace internal
  {
    namespace QuadratureGeneratorImplementation
    {
      template <int dim, int spacedim>
      class QGenerator;
    }
  } // namespace internal


  /**
   * Struct storing settings for the QuadratureGenerator class.
   */
  struct AdditionalQGeneratorData
  {
    /**
     * Constructor.
     */
    AdditionalQGeneratorData(
      const unsigned int n_allowed_splits              = 2,
      const double       lower_bound_implicit_function = 1e-11,
      const double       min_distance_between_roots    = 1e-12,
      const double       limit_to_be_definite          = -1e-11,
      const double       root_finder_tolerance         = 1e-12,
      const unsigned int n_allowed_root_finder_splits  = 2);

    /**
     * The number of times we are allowed to split the incoming box
     * and recurse on each child.
     */
    unsigned int n_allowed_splits;

    /**
     * For a level set function, $\psi$, the implicit function theorem states
     * that it is possible to write one of the coordinates $x_i$ as a function
     * of the others if
     *
     * $|\frac{\partial \psi}{\partial x_i}| > 0$.
     *
     * In practice, it might happend the bound we have for the expression in
     * the left-hand side is only floating-point close to zero.
     *
     * This constant is a safety margin, $C$, that states that the implicit
     * function theorem can be used when
     *
     * $|\frac{\partial \psi}{\partial x_i}| > C$
     *
     * Thus this constant must be non-negative.
     */
    double lower_bound_implicit_function;

    /**
     * If two roots are closer to each other than this distance they are
     * merged to one.
     */
    double min_distance_between_roots;

    /**
     * A constant, $C$, controlling when a level set function, $\psi$, is
     * considered positive or negative definite:
     *
     * $\psi(x) >  C \RightArrow \text{Positive definite}$,
     * $\psi(x) < -C \RightArrow \text{Negative definite}$.
     */
    double limit_to_be_definite;

    /**
     * Tolerance for convergence of the underlying root finder.
     */
    double root_finder_tolerance;

    /**
     * The number of times the underlying rootfinder is allowed to split
     * an interval, while trying to find multiple roots.
     */
    unsigned int n_allowed_root_finder_splits;
  };



  /**
   * Creates immersed quadrature rules over a BoundingBox,
   * $K \subset \mathbb{R}^{dim}$, when the domain is described by a level set
   * function, $\psi$.
   *
   * This class creates quadrature rules for the intersections between the box and
   * the three different regions defined by the level set function. That is, it
   * creates quadrature rules to integrate over the following regions
   *
   * $\hat{K} \cap \{x \in \mathbb{R}^{dim} : \psi(x) < 0 \}$,
   * $\hat{K} \cap \{x \in \mathbb{R}^{dim} : \psi(x) > 0 \}$,
   * $\hat{K} \cap \{x \in \mathbb{R}^{dim} : \psi(x) = 0 \}$.
   *
   * @image html immersed_quadratures.svg
   *
   * When working with level set functions, the most common is to describe a
   * domain, $\Omega$, as
   *
   * $\Omega = \{ x \in \mathbb{R}^{dim} : \psi(x) < 0 \}$.
   *
   * Given this, we shall use the name convention that $\{x : \psi(x) < 0 \}$
   * is the "inside" region (i.e. inside $\Omega$), $\{x : \psi(x) > 0 \}$ is
   * the "outside" region and $\{x : \psi(x) = 0 \}$ is the "surface".
   *
   * The number of quadrature points in the constructed quadratures will vary
   * depending on how the level set intersects the box. More quadrature points
   * will be created if the intersection is "bad". For example if the curve
   * defined by the zero-countour has a high curvature.
   *
   * The underlying algorithm use a 1-dimensional quadrature rule as base for
   * creating the immersed quadrature rules. The constructor takes a
   * hp::QCollection<1> of and one can select which 1D-quadrature in the
   * collection should be used through the function set_1D_quadrature(). Any
   * 1D-quadrature (over the interval $[0,1]$) can be used but Gauss-Legendre
   * quadrature is recommended. QGauss<1> of order p gives errors proportional
   * to $h^p$.
   *
   * A detailed description of the underlying algorithm can be found in
   * "High-Order %Quadrature Methods for Implicitly Defined Surfaces and
   * Volumes in Hyperrectangles, R. I. Saye, SIAM J. Sci. Comput., 37(2), <a
   * href="http://www.dx.doi.org/10.1137/140966290">
   * doi:10.1137/140966290</a>". The implemented in this class has some minor
   * modifications compared to the algorithm description in the paper.
   */
  template <int dim>
  class QuadratureGenerator
  {
  public:
    using AdditionalData = AdditionalQGeneratorData;

    /**
     * Constructor. Each Quadrature<1> in @p quadratures1D can be chosen as base
     * for generating the immersed quadrature rules.
     *
     * @note: It is important that each 1D-quadrature rule in the
     * hp::QCollection does not contain the points 0 and 1.
     */
    QuadratureGenerator(
      const hp::QCollection<1> &quadratures1D,
      const AdditionalData &    additional_data = AdditionalData());

    /**
     * Construct immersed quadratures rules for the incoming level set
     * function over the BoundingBox.
     *
     * To get the constructed quadratures, use the functions
     * get_inside_quadrature(),
     * get_outside_quadrature(),
     * get_surface_quadrature().
     */
    void
    create_quadrature(const std::shared_ptr<Function<dim>> &level_set,
                      const BoundingBox<dim> &              box);

    /*
     * Return a quadrature rule for the region
     * $K \cap \{x : psi(x) < 0 \}$
     * created in the previous call to create_quadrature().
     * Here, $K$ is BoundingBox passed to create_quadrature().
     */
    const Quadrature<dim> &
    get_inside_quadrature() const;


    /*
     * Return a quadrature rule for the region
     * $K \cap \{x : psi(x) > 0 \}$
     * created in the previous call to create_quadrature().
     * Here, $K$ is BoundingBox passed to create_quadrature().
     */
    const Quadrature<dim> &
    get_outside_quadrature() const;

    /*
     * Return a quadrature rule for the region
     * $K \cap \{x : psi(x) = 0 \}$
     * created in the previous call to create_quadrature().
     * Here, $K$ is BoundingBox passed to create_quadrature().
     */
    const ImmersedSurfaceQuadrature<dim> &
    get_surface_quadrature() const;

    /**
     * Set which 1D-quadrature in the collection passed to the constructor
     * should be used to create the immersed quadratures.
     */
    void
    set_1D_quadrature(const unsigned int q_index);

  private:
    /**
     * QuadratureGenerator is mainly used to start up the recursive
     * algorithm. This is the object that actually generates the quadratures.
     */
    internal::QuadratureGeneratorImplementation::QGenerator<dim, dim>
      q_generator;
  };

  namespace internal
  {
    namespace QuadratureGeneratorImplementation
    {
      /**
       * A class that attempts to find multiple distinct roots of a function,
       * $f(x)$, over an interval, $[l, r]$. This is done as follows. If there
       * is a sign change in function value between the interval end points,
       * we solve for the root. If there is no sign change, we attempt to
       * bound the function value away from zero on $[a, b]$, to conclude that
       * no roots exist. If we can't exclude that there are roots, we split
       * the interval in two: $[l, (r + l) / 2]$, $[(r + l) / 2, r]$, and use
       * the same algorithm recursively on each interval. This means that we
       * can typically find 2 distinct roots, but not 3.
       *
       * The bounds on the functions value are estimated using the function
       * taylor_estimate_function_bounds, which approximates the function as a
       * second order Taylor-polynomial around the interval midpoint.
       * When we have a sign change on an interval, this class uses
       * boost::math::tools::toms748_solve for finding roots .
       */
      class RootFinder
      {
      public:
        /**
         * Struct storing settings for the RootFinder class.
         */
        struct AdditionalData
        {
          /**
           * Constructor.
           */
          AdditionalData(const double       tolerance           = 1e-12,
                         const unsigned int max_recursion_depth = 2,
                         const unsigned int max_iterations      = 500);

          /**
           * The tolerance in the stopping criteria for the underlying root
           * finding algorithm boost::math::tools::toms748_solve.
           */
          double tolerance;

          /**
           * The number of times we are allowed to split the interval where we
           * seek roots.
           */
          unsigned int max_recursion_depth;

          /**
           * The maximum number of iterations in
           * boost::math::tools::toms748_solve.
           */
          unsigned int max_iterations;
        };


        /**
         * Constructor.
         */
        RootFinder(const AdditionalData &data = AdditionalData());

        /**
         * For each of the incoming @p functions, attempt to find the roots over
         * the interval defined by @p interval and add these to @p roots.
         * The returned roots will be sorted increasingly: $x_0 < x_1 < ...$
         * and non-unique roots (with respect to the tolerance in
         * AdditionalData) will be removed.
         */
        void
        find_roots(const std::vector<std::shared_ptr<Function<1>>> &functions,
                   const BoundingBox<1> &                           interval,
                   std::vector<double> &                            roots);

      private:
        /**
         * Attempt to find the roots of the @p function over the interval defined by
         * @p interval and add these to @p roots. @p recursion_depth holds the number
         * of times this function has been called recursively.
         */
        void
        find_roots(const Function<1> &   function,
                   const BoundingBox<1> &interval,
                   const unsigned int    recursion_depth,
                   std::vector<double> & roots);

        const AdditionalData additional_data;
      };


      /*
       * This is just a Quadrature which has push_back method for adding a
       * point with an associated weight.
       *
       * Since we build the quadrature rules in step-wise fashion,
       * it's easier to use this class than to pass around two vectors:
       * std::vector<Point<dim>>,
       * std::vector<double>.
       * Further, two std::vectors could accidentally end up with different
       * sizes. Using push_back we make sure that the number of points and
       * weights are the same.
       */
      template <int dim>
      class ExtendableQuadrature : public Quadrature<dim>
      {
      public:
        /**
         * Constructor, creates an empty quadrature rule with no
         * points.
         */
        ExtendableQuadrature() = default;

        /**
         * Constructor, copies the incoming Quadrature.
         */
        ExtendableQuadrature(const Quadrature<dim> &quadrature);

        /**
         * Add a point with an associated weight to the quadrature.
         */
        void
        push_back(const Point<dim> &point, const double weight);
      };


      /**
       * Type that describes the definiteness of a function over some interval
       * or region.
       */
      enum class Definiteness
      {
        NEGATIVE,
        POSITIVE,
        INDEFINITE
      };


      /**
       * Class that stores the quadrature rules over the different regions of
       * a cell.
       */
      template <int dim>
      class ImmersedQuadratures : public Subscriptor
      {
      public:
        /**
         * Returns one of the "bulk" quadratures of this class,
         * using the following mapping:
         *
         * Definiteness -> Quadrature
         *   POSITIVE   ->  outside
         *   NEGATIVE   ->  inside
         *  INDEFINITE  -> indefinite
         *
         * This corresponds to the signs that the level set functions have,
         * on the domain that the quadrature integrates over.
         */
        ExtendableQuadrature<dim> &
        quadrature_by_definiteness(const Definiteness definiteness);

        /*
         * Quadrature for the region $\{x : psi(x) < 0 \}$ of the box.
         */
        ExtendableQuadrature<dim> inside;

        /*
         * Quadrature for the region $\{x : psi(x) > 0 \}$ of the box.
         */
        ExtendableQuadrature<dim> outside;

        /*
         * In the algorithm used by QuadratureGenerator we go down recursively
         * throught the dimensions 3 -> 2 -> 1. When going down once create
         * two new level set functions by restricting the function the top and
         * bottom faces of the BoundingBox, in some direction.
         *
         * This is a quadrature for a region that is "indefinite" in the sense
         * that the restrictions have different sign over the region.
         */
        ExtendableQuadrature<dim> indefinite;

        /*
         * Quadrature for the region $\{x : psi(x) = 0 \}$ of the box.
         */
        ImmersedSurfaceQuadrature<dim> surface;
      };


      template <int dim, int spacedim>
      class UpThroughDimensionCreator
      {
      public:
        UpThroughDimensionCreator(
          ImmersedQuadratures<dim> &      immersed_quadratures,
          const hp::QCollection<1> &      q_collection1D,
          const AdditionalQGeneratorData &additional_data);

        void
        create_quadrature(
          const std::vector<std::shared_ptr<Function<dim>>> &level_sets,
          const BoundingBox<dim> &                           box,
          const Quadrature<dim - 1> &                        low_dim_quadrature,
          const unsigned int height_function_direction);

        /**
         * Set which 1D-quadrature in the collection passed to the constructor
         * should be used to create the immersed quadratures.
         */
        void
        set_1D_quadrature(const unsigned int q_index);

      private:
        void
        create_surface_point(
          const Point<dim - 1> &                             point,
          const double                                       weight,
          const std::vector<std::shared_ptr<Function<dim>>> &level_sets,
          const BoundingBox<dim> &                           box);

        /**
         * Pointer to the collection of quadratures that this class adds
         * quadrature points to.
         */
        const SmartPointer<ImmersedQuadratures<dim>> immersed_quadratures;

        /**
         * One dimensional quadrature rules used to create the immersed
         * quadratures.
         */
        const SmartPointer<const hp::QCollection<1>> q_collection1D;

        const AdditionalQGeneratorData additional_data;

        /**
         * Which quadrature rule in the above collection that is used to
         * create the immersed quadrature rules.
         */
        unsigned int q_index;

        /**
         * The height function direction in the last call to
         * create_quadrature.
         */
        unsigned int height_function_direction;

        std::vector<std::shared_ptr<Function<1>>> restrictions1D;

        RootFinder          root_finder;
        std::vector<double> roots;
      };


      /*
       * Base class for the class QGenerator<dim, spacedim> and the
       * one-dimensional specialization QGenerator<1, spacedim>.
       */
      template <int dim, int spacedim>
      class QGeneratorBase : public Subscriptor
      {
      public:
        virtual void
        create_quadrature(
          const std::vector<std::shared_ptr<Function<dim>>> &restrictions,
          const BoundingBox<dim> &                           hypercube) = 0;

        /**
         * Clear the quadratures created by the previous call to
         * create_quadrature().
         */
        void
        clear_quadratures();

        /**
         * Return the quadratures created by the previous call to
         * create_quadrature().
         */
        const ImmersedQuadratures<dim> &
        get_quadratures() const;

      protected:
        /**
         * Quadratures that the derived class should create.
         */
        ImmersedQuadratures<dim> immersed_quadratures;
      };


      template <int dim, int spacedim>
      class QGenerator : public QGeneratorBase<dim, spacedim>
      {
      public:
        /**
         * Constructor.
         */
        QGenerator(const hp::QCollection<1> &      q_collection1D,
                   const AdditionalQGeneratorData &additional_data);

        /**
         * Creates quadrature points over the incoming @p box.
         * To get these call get_quadratures.
         */
        void
        create_quadrature(
          const std::vector<std::shared_ptr<Function<dim>>> &level_sets,
          const BoundingBox<dim> &                           box) override;

        /**
         * Set which 1D-quadrature in the collection passed to the constructor
         * should be used to create the immersed quadratures.
         */
        void
        set_1D_quadrature(const unsigned int q_index);

        /**
         * Clear the quadratures created by the previous call to
         * create_quadrature().
         */
        void
        clear_quadratures();

      private:
        /**
         * Restricts the incoming level set functions to the top and bottom of
         * the incoming box (w.r.t @p height_function_direction). Then call the
         * lower dimensional QGenerator with the cross section of the box
         * to generate the lower dimensional immersed quadrature rules.
         */
        void
        create_low_dim_quadratures(
          const unsigned int height_function_direction,
          const std::vector<std::shared_ptr<Function<dim>>> &level_sets,
          const BoundingBox<dim> &                           box);

        void
        create_high_dim_quadratures(
          const unsigned int height_function_direction,
          const std::vector<std::shared_ptr<Function<dim>>> &level_sets,
          const BoundingBox<dim> &                           box);

        /**
         * Defines a stopping criteria for when we are allowed to split a box
         * into it's children and run the algorithm on each child. Determines
         * this by checking the size of the incoming box.
         */
        bool
        allowed_to_split_cube(const BoundingBox<dim> &box) const;

        void
        split_box_and_recurse(
          const std::vector<std::shared_ptr<Function<dim>>> &level_sets,
          const BoundingBox<dim> &                           box);

        /**
         * Uses the midpoint-method to create a quadrature over the @param box.
         * That is, add a single quadrature point at the center of the box
         * with weight corresponding to the volume of the box.
         *
         * If all the @param level_sets are negative/positive at the center of the box,
         * the point is added to ImmersedQuadratures::inside/outside.
         * Otherwise it is added to ImmersedQuadratures::indefinite.
         */
        void
        use_midpoint_method(
          const std::vector<std::shared_ptr<Function<dim>>> &level_sets,
          const BoundingBox<dim> &                           box);


        const AdditionalQGeneratorData additional_data;

        /**
         * The same algorithm as this, but creating immersed quadratures
         * in one dimension lower.
         */
        QGenerator<dim - 1, spacedim> low_dim_algorithm;

        UpThroughDimensionCreator<dim, spacedim> up_through_dimension_creator;

        /**
         * Which 1D-quadrature in the collection we should use to generate
         * the immersed quadrature.
         */
        unsigned int q_index;

        /**
         * Index of the quadrature in q_collection1D that should use to
         * generate the immersed quadrature rules.
         */
        const SmartPointer<const hp::QCollection<1>> q_collection1D;

        /**
         * Stores tensor products of each of the Quadrature<1>'s in
         * q_collection1D.
         */
        hp::QCollection<dim> tensor_products;
      };


      /**
       * The one dimensional base case of the recursive algorithm in
       * QGenerator<dim, spacedim>. This class creates one-dimensional
       * quadrature points over the interval defined by the BoundingBox<1>
       * sent into create_quadrature() and stores these internally in one of
       * the quadratures in ImmersedQuadratures.
       *
       * Let $L$ and $R$ be the lower and upper bounds of the one-dimensional
       * BoundingBox. This interval is partitioned into $[x_0, x_1, ..., x_n]$
       * where $x_0$, $x_n$ are the interval end points and the remaining $x_i$
       * are the sorted roots in the interval $[L, R]$. In each interval,
       * $[x_i, x_{i+1}]$, quadrature points are distributed according to the
       * 1D-quadrature rule. These points are added to one of the regions of
       * ImmersedQuadratures. Which quadrature the points are added to are
       * determined from the sign of the Function<1>'s at the interval midpoint.
       * If all are restrictions are negative/positive the points are added to
       * ImmersedQuadratures::inside/ImmersedQuadratures::outside.
       * If some of the restrictions are positive and some are negative the
       * points are added to ImmersedQuadratures::indefinite.
       *
       * If spacedim = 1 the points $[x_1, x_{n-1}]$ are also added as surface
       * quadrature points with weight 1 to ImmersedQuadratures::surface.
       */
      template <int spacedim>
      class QGenerator<1, spacedim> : public QGeneratorBase<1, spacedim>
      {
      public:
        QGenerator(const hp::QCollection<1> &      quadratures1D,
                   const AdditionalQGeneratorData &additional_data);

        /**
         * Creates quadrature points over interval defined by the incoming @p box
         * and adds these quadrature points to the internally stored
         * ImmersedQuadratures. These quadratures can then be obtained by
         * calling get_quadratures.
         *
         * This is done by partitioning the interval between the lower and upper bound of @p box
         * by the roots of the @p restrictions. For each interval in the partioning,
         * we then distributes quadrature points between the interval start
         * and the end according to the one-dimensional quadrature rule. Adds
         * these points to one of the regions of ImmersedQuadratures. See
         * detailed description in class description.
         */
        void
        create_quadrature(
          const std::vector<std::shared_ptr<Function<1>>> &restrictions,
          const BoundingBox<1> &                           box) override;

        /**
         * Set which 1D-quadrature in the collection passed to the constructor
         * should be used to create the immersed quadratures.
         */
        void
        set_1D_quadrature(const unsigned int q_index);

      private:
        /**
         * Adds the point defined by coordinate to the surface quadrature of
         * ImmersedQuadrature with unit weight.
         */
        void
        create_surface_points(
          const std::vector<std::shared_ptr<Function<1>>> &restrictions);

        const AdditionalQGeneratorData additional_data;

        /**
         * Index of the quadrature in q_collection1D that should use to
         * generate the immersed quadrature rules.
         */
        unsigned int q_index;

        /**
         * 1D-quadrature rules that can be chosen as base for creating the
         * immersed quadrature rules.
         */
        const SmartPointer<const hp::QCollection<1>> q_collection1D;

        /**
         * Class used to find the roots of the functions passed to
         * create_quadrature().
         */
        RootFinder root_finder;

        /**
         * Roots of the functions passed to create_quadrature().
         */
        std::vector<double> roots;

        /*
         * This would be the height-function direction in higher dimensions,
         * but in 1D there is only one coordinate direction.
         */
        const unsigned int direction = 0;

        /*
         * To reuse the distribute_points_between_roots()-function
         * we need a zero-dimensional quadrature point with unit weight.
         */
        const Point<0> zero_dim_point;
        const double   unit_weight = 1;
      };

      /**
       * Take the tensor product between (point, weight) and @p quadrature1D
       * scaled over [start, end] and add the resulting dim-dimensional
       * quadrature points to @p quadrature.
       *
       * @p component_in_dim specifies which dim-dimensional coordinate
       * quadrature1D should be written to.
       */
      template <int dim>
      void
      quadrature_from_point(const Quadrature<1> &      quadrature1D,
                            const Point<dim - 1> &     point,
                            const double               weight,
                            const unsigned int         component_in_dim,
                            const double               start,
                            const double               end,
                            ExtendableQuadrature<dim> &quadrature);


      /**
       * Checks the sign of the incoming Functions at the incoming point and
       * returns Definiteness::POSITIVE/Definiteness::NEGATIVE if all the
       * functions are positive/negative at the point, otherwise returns
       * Definiteness::INDEFINITE.
       */
      template <int dim>
      Definiteness
      pointwise_definiteness(
        const std::vector<std::shared_ptr<Function<dim>>> &functions,
        const Point<dim> &                                 point);


      /**
       * A struct storing the bounds on the function value and bounds
       * on each component of the gradient.
       */
      template <int dim>
      struct FunctionBounds
      {
      public:
        /**
         * Lower and upper bounds on the functions value.
         */
        std::pair<double, double> value;

        /**
         * Lower and upper bounds on each component of the gradient.
         */
        std::array<std::pair<double, double>, dim> gradient;
      };


      /**
       * Returns the max/min bounds on the value, taken over all the entries
       * in the incoming vector of FunctionBounds.
       */
      template <int dim>
      std::pair<double, double>
      find_extreme_values(
        const std::vector<FunctionBounds<dim>> &all_function_bounds);


      /**
       * Returns the componentwise minimum of min_abs_gradient
       * of all the incoming FunctionBounds.
       *
       * Currently, this function ignores FunctionBounds which corresponds
       * to functions which are negative/positive definite. If all incoming
       * bounds are definite the returned optional is non-set.
       */
      template <int dim>
      std_cxx17::optional<Tensor<1, dim>>
      min_of_all_min_abs_grad(
        const std::vector<FunctionBounds<dim>> &all_function_bounds);


      /**
       * Data representing the best choice of height-function direction,
       * which is returned by the function find_best_height_direction.
       *
       * This data consists of a coordinate direction
       *
       * $i \in \{0, ..., dim - 1 \}$,
       *
       * and lower bound on the absolute value of the derivative of some
       * associated function, f, taken in the above coordinate direction. That
       * is, a bound $C$ such that
       *
       * $|\frac{\partial f}{\partial x_i}| > C$,
       *
       * holding over some subset of $\mathbb{R}^{dim}$.
       */
      struct HeightDirectionData
      {
        /**
         * Constructor. Initializes the direction to invalid_unsigned_int and
         * the bound to 0.
         */
        HeightDirectionData();


        /**
         * The height-function direction, described above.
         */
        unsigned int direction;

        /**
         * The lower bound on $|\frac{\partial f}{\partial x_i}|$, described
         * above.
         */
        double min_abs_dfdx;
      };


      /**
       * Given the FunctionBounds for a number of functions $\{ f_j \}_j$,
       * finds the best choice of height function direction,
       * where "best" is meant in the sense of the implicit function theorem.
       * We want to find a direction, $i$, such that all the functions fulfill
       *
       * $\frac{\partial f_j}{\partial x_i} \neq 0  \forall j$.
       *
       * This function finds a coordindate direction, i, and a lower bound C,
       * such that
       *
       * $i = arg max_{k} min_{j} |\frac{\partial f_j}{\partial x_k}|$
       * $C =     max_{k} min_{j} |\frac{\partial f_j}{\partial x_k}|$
       *
       * Note that the returned lower bound might be zero. This means that no
       * suitable height function direction exists.
       */
      template <int dim>
      std_cxx17::optional<HeightDirectionData>
      find_best_height_direction(
        const std::vector<FunctionBounds<dim>> &all_function_bounds);

    } // namespace QuadratureGeneratorImplementation
  }   // namespace internal

} // namespace NonMatching
DEAL_II_NAMESPACE_CLOSE

#endif /* dealii_non_matching_quadrature_generator_h */
