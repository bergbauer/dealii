// ---------------------------------------------------------------------
//
// Copyright (C) 2011 - 2021 by the deal.II authors
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


#ifndef dealii_matrix_free_fe_point_h
#define dealii_matrix_free_fe_point_h

#include <deal.II/base/config.h>

#include <deal.II/base/aligned_vector.h>
#include <deal.II/base/exceptions.h>
#include <deal.II/base/quadrature.h>
#include <deal.II/base/template_constraints.h>
#include <deal.II/base/thread_local_storage.h>
#include <deal.II/base/vectorization.h>

#include <deal.II/dofs/dof_handler.h>

#include <deal.II/fe/fe.h>
#include <deal.II/fe/mapping.h>
#include <deal.II/fe/mapping_q1.h>

#include <deal.II/hp/dof_handler.h>
#include <deal.II/hp/mapping_collection.h>
#include <deal.II/hp/q_collection.h>

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/block_vector_base.h>
#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/lac/vector_operation.h>

#include <deal.II/matrix_free/dof_info.h>
#include <deal.II/matrix_free/mapping_info.h>
#include <deal.II/matrix_free/shape_info.h>
#include <deal.II/matrix_free/task_info.h>
#include <deal.II/matrix_free/type_traits.h>

#include <cstdlib>
#include <limits>
#include <list>
#include <memory>


DEAL_II_NAMESPACE_OPEN



/**
 * This class collects all the data that is stored for the matrix free
 * implementation.
 *
 * For details on usage of this class, see the description of FEPointEvaluation
 * or the
 * @ref matrixfree "matrix-free module".
 *
 * @ingroup matrixfree
 */

template <int dim,
          typename Number              = double,
          typename VectorizedArrayType = VectorizedArray<Number>>
class MatrixFreeFEPoint : public Subscriptor
{
  static_assert(
    std::is_same<Number, typename VectorizedArrayType::value_type>::value,
    "Type of Number and of VectorizedArrayType do not match.");

public:
  /**
   * An alias for the underlying number type specified by the template
   * argument.
   */
  using value_type            = Number;
  using vectorized_value_type = VectorizedArrayType;

  /**
   * The dimension set by the template argument `dim`.
   */
  static constexpr unsigned int dimension = dim;

  /**
   * Collects the options for initialization of the MatrixFree class. The
   * first parameter specifies the MPI communicator to be used, the second the
   * parallelization options in shared memory (task-based parallelism, where
   * one can choose between no parallelism and three schemes that avoid that
   * cells with access to the same vector entries are accessed
   * simultaneously), the third with the block size for task parallel
   * scheduling, the fourth the update flags that should be stored by this
   * class.
   *
   * The fifth parameter specifies the level in the triangulation from which
   * the indices are to be used. If the level is set to
   * numbers::invalid_unsigned_int, the active cells are traversed, and
   * otherwise the cells in the given level. This option has no effect in case
   * a DoFHandler is given.
   *
   * The parameter @p initialize_plain_indices indicates whether the DoFInfo
   * class should also allow for access to vectors without resolving
   * constraints.
   *
   * The two parameters `initialize_indices` and `initialize_mapping` allow
   * the user to disable some of the initialization processes. For example, if
   * only the scheduling that avoids touching the same vector/matrix indices
   * simultaneously is to be found, the mapping needs not be
   * initialized. Likewise, if the mapping has changed from one iteration to
   * the next but the topology has not (like when using a deforming mesh with
   * MappingQEulerian), it suffices to initialize the mapping only.
   *
   * The two parameters `cell_vectorization_categories` and
   * `cell_vectorization_categories_strict` control the formation of batches
   * for vectorization over several cells. It is used implicitly when working
   * with hp-adaptivity but can also be useful in other contexts, such as in
   * local time stepping where one would like to control which elements
   * together form a batch of cells. The array `cell_vectorization_categories`
   * is accessed by the number given by cell->active_cell_index() when working
   * on the active cells with `mg_level` set to `numbers::invalid_unsigned_int`
   * and by cell->index() for the level cells. By default, the different
   * categories in `cell_vectorization_category` can be mixed and the algorithm
   * is allowed to merge lower category numbers with the next higher categories
   * if it is necessary inside the algorithm, in order to avoid partially
   * filled SIMD lanes as much as possible. This gives a better utilization of
   * the vectorization but might need special treatment, in particular for
   * face integrals. If set to @p true, the algorithm will instead keep
   * different categories separate and not mix them in a single vectorized
   * array.
   */
  struct AdditionalData
  {
    /**
     * Provide the type of the surrounding MatrixFree class.
     */
    using MatrixFreeType = MatrixFreeFEPoint<dim, Number, VectorizedArrayType>;

    /**
     * Collects options for task parallelism. See the documentation of the
     * member variable MatrixFree::AdditionalData::tasks_parallel_scheme for a
     * thorough description.
     */
    enum TasksParallelScheme
    {
      /**
       * Perform application in serial.
       */
      none = internal::MatrixFreeFunctions::TaskInfo::none,
      /**
       * Partition the cells into two levels and afterwards form chunks.
       */
      partition_partition =
        internal::MatrixFreeFunctions::TaskInfo::partition_partition,
      /**
       * Partition on the global level and color cells within the partitions.
       */
      partition_color =
        internal::MatrixFreeFunctions::TaskInfo::partition_color,
      /**
       * Use the traditional coloring algorithm: this is like
       * TasksParallelScheme::partition_color, but only uses one partition.
       */
      color = internal::MatrixFreeFunctions::TaskInfo::color
    };

    /**
     * Constructor for AdditionalData.
     */
    AdditionalData(
      const TasksParallelScheme tasks_parallel_scheme = partition_partition,
      const unsigned int        tasks_block_size      = 0,
      const UpdateFlags         mapping_update_flags  = update_gradients |
                                               update_JxW_values,
      const UpdateFlags  mapping_update_flags_boundary_faces = update_default,
      const UpdateFlags  mapping_update_flags_inner_faces    = update_default,
      const UpdateFlags  mapping_update_flags_faces_by_cells = update_default,
      const unsigned int mg_level            = numbers::invalid_unsigned_int,
      const bool         store_plain_indices = true,
      const bool         initialize_indices  = true,
      const bool         initialize_mapping  = true,
      const bool         overlap_communication_computation    = true,
      const bool         hold_all_faces_to_owned_cells        = false,
      const bool         cell_vectorization_categories_strict = false,
      const bool         allow_ghosted_vectors_in_loops       = true)
      : tasks_parallel_scheme(tasks_parallel_scheme)
      , tasks_block_size(tasks_block_size)
      , mapping_update_flags(mapping_update_flags)
      , mapping_update_flags_boundary_faces(mapping_update_flags_boundary_faces)
      , mapping_update_flags_inner_faces(mapping_update_flags_inner_faces)
      , mapping_update_flags_faces_by_cells(mapping_update_flags_faces_by_cells)
      , mg_level(mg_level)
      , store_plain_indices(store_plain_indices)
      , initialize_indices(initialize_indices)
      , initialize_mapping(initialize_mapping)
      , overlap_communication_computation(overlap_communication_computation)
      , hold_all_faces_to_owned_cells(hold_all_faces_to_owned_cells)
      , cell_vectorization_categories_strict(
          cell_vectorization_categories_strict)
      , allow_ghosted_vectors_in_loops(allow_ghosted_vectors_in_loops)
      , communicator_sm(MPI_COMM_SELF)
    {}

    /**
     * Copy constructor.
     */
    AdditionalData(const AdditionalData &other)
      : tasks_parallel_scheme(other.tasks_parallel_scheme)
      , tasks_block_size(other.tasks_block_size)
      , mapping_update_flags(other.mapping_update_flags)
      , mapping_update_flags_boundary_faces(
          other.mapping_update_flags_boundary_faces)
      , mapping_update_flags_inner_faces(other.mapping_update_flags_inner_faces)
      , mapping_update_flags_faces_by_cells(
          other.mapping_update_flags_faces_by_cells)
      , mg_level(other.mg_level)
      , store_plain_indices(other.store_plain_indices)
      , initialize_indices(other.initialize_indices)
      , initialize_mapping(other.initialize_mapping)
      , overlap_communication_computation(
          other.overlap_communication_computation)
      , hold_all_faces_to_owned_cells(other.hold_all_faces_to_owned_cells)
      , cell_vectorization_category(other.cell_vectorization_category)
      , cell_vectorization_categories_strict(
          other.cell_vectorization_categories_strict)
      , allow_ghosted_vectors_in_loops(other.allow_ghosted_vectors_in_loops)
      , communicator_sm(other.communicator_sm)
    {}

    /**
     * Copy assignment.
     */
    AdditionalData &
    operator=(const AdditionalData &other)
    {
      tasks_parallel_scheme = other.tasks_parallel_scheme;
      tasks_block_size      = other.tasks_block_size;
      mapping_update_flags  = other.mapping_update_flags;
      mapping_update_flags_boundary_faces =
        other.mapping_update_flags_boundary_faces;
      mapping_update_flags_inner_faces = other.mapping_update_flags_inner_faces;
      mapping_update_flags_faces_by_cells =
        other.mapping_update_flags_faces_by_cells;
      mg_level            = other.mg_level;
      store_plain_indices = other.store_plain_indices;
      initialize_indices  = other.initialize_indices;
      initialize_mapping  = other.initialize_mapping;
      overlap_communication_computation =
        other.overlap_communication_computation;
      hold_all_faces_to_owned_cells = other.hold_all_faces_to_owned_cells;
      cell_vectorization_category   = other.cell_vectorization_category;
      cell_vectorization_categories_strict =
        other.cell_vectorization_categories_strict;
      allow_ghosted_vectors_in_loops = other.allow_ghosted_vectors_in_loops;
      communicator_sm                = other.communicator_sm;

      return *this;
    }

    /**
     * Set the scheme for task parallelism. There are four options available.
     * If set to @p none, the operator application is done in serial without
     * shared memory parallelism. If this class is used together with MPI and
     * MPI is also used for parallelism within the nodes, this flag should be
     * set to @p none. The default value is @p partition_partition, i.e. we
     * actually use multithreading with the first option described below.
     *
     * The first option @p partition_partition is to partition the cells on
     * two levels in onion-skin-like partitions and forming chunks of
     * tasks_block_size after the partitioning. The partitioning finds sets of
     * independent cells that enable working in parallel without accessing the
     * same vector entries at the same time.
     *
     * The second option @p partition_color is to use a partition on the
     * global level and color cells within the partitions (where all chunks
     * within a color are independent). Here, the subdivision into chunks of
     * cells is done before the partitioning, which might give worse
     * partitions but better cache performance if degrees of freedom are not
     * renumbered.
     *
     * The third option @p color is to use a traditional algorithm of coloring
     * on the global level. This scheme is a special case of the second option
     * where only one partition is present. Note that for problems with
     * hanging nodes, there are quite many colors (50 or more in 3D), which
     * might degrade parallel performance (bad cache behavior, many
     * synchronization points).
     *
     * @note Threading support is currently experimental for the case inner
     * face integrals are performed and it is recommended to use MPI
     * parallelism if possible. While the scheme has been verified to work
     * with the `partition_partition` option in case of usual DG elements, no
     * comprehensive tests have been performed for systems of more general
     * elements, like combinations of continuous and discontinuous elements
     * that add face integrals to all terms.
     */
    TasksParallelScheme tasks_parallel_scheme;

    /**
     * Set the number of so-called macro cells that should form one
     * partition. If zero size is given, the class tries to find a good size
     * for the blocks based on MultithreadInfo::n_threads() and the number of
     * cells present. Otherwise, the given number is used. If the given number
     * is larger than one third of the number of total cells, this means no
     * parallelism. Note that in the case vectorization is used, a macro cell
     * consists of more than one physical cell.
     */
    unsigned int tasks_block_size;

    /**
     * This flag determines the mapping data on cells that is cached. This
     * class can cache data needed for gradient computations (inverse
     * Jacobians), Jacobian determinants (JxW), quadrature points as well as
     * data for Hessians (derivative of Jacobians). By default, only data for
     * gradients and Jacobian determinants times quadrature weights, JxW, are
     * cached. If quadrature points or second derivatives are needed, they
     * must be specified by this field (even though second derivatives might
     * still be evaluated on Cartesian cells without this option set here,
     * since there the Jacobian describes the mapping completely).
     */
    UpdateFlags mapping_update_flags;

    /**
     * This flag determines the mapping data on boundary faces to be
     * cached. Note that MatrixFree uses a separate loop layout for face
     * integrals in order to effectively vectorize also in the case of hanging
     * nodes (which require different subface settings on the two sides) or
     * some cells in the batch of a VectorizedArray of cells that are adjacent
     * to the boundary and others that are not.
     *
     * If set to a value different from update_general (default), the face
     * information is explicitly built. Currently, MatrixFree supports to
     * cache the following data on faces: inverse Jacobians, Jacobian
     * determinants (JxW), quadrature points, data for Hessians (derivative of
     * Jacobians), and normal vectors.
     *
     * @note In order to be able to perform a `boundary_operation` in the
     * MatrixFree::loop(), this field must be set to a value different from
     * UpdateFlags::update_default.
     */
    UpdateFlags mapping_update_flags_boundary_faces;

    /**
     * This flag determines the mapping data on interior faces to be
     * cached. Note that MatrixFree uses a separate loop layout for face
     * integrals in order to effectively vectorize also in the case of hanging
     * nodes (which require different subface settings on the two sides) or
     * some cells in the batch of a VectorizedArray of cells that are adjacent
     * to the boundary and others that are not.
     *
     * If set to a value different from update_general (default), the face
     * information is explicitly built. Currently, MatrixFree supports to
     * cache the following data on faces: inverse Jacobians, Jacobian
     * determinants (JxW), quadrature points, data for Hessians (derivative of
     * Jacobians), and normal vectors.
     *
     * @note In order to be able to perform a `face_operation`
     * in the MatrixFree::loop(), this field must be set to a value different
     * from UpdateFlags::update_default.
     */
    UpdateFlags mapping_update_flags_inner_faces;

    /**
     * This flag determines the mapping data for faces in a different layout
     * with respect to vectorizations. Whereas
     * `mapping_update_flags_inner_faces` and
     * `mapping_update_flags_boundary_faces` trigger building the data in a
     * face-centric way with proper vectorization, the current data field
     * attaches the face information to the cells and their way of
     * vectorization. This is only needed in special situations, as for
     * example for block-Jacobi methods where the full operator to a cell
     * including its faces are evaluated. This data is accessed by
     * <code>FEFaceEvaluation::reinit(cell_batch_index,
     * face_number)</code>. However, currently no coupling terms to neighbors
     * can be computed with this approach because the neighbors are not laid
     * out by the VectorizedArray data layout with an
     * array-of-struct-of-array-type data structures.
     *
     * Note that you should only compute this data field in case you really
     * need it as it more than doubles the memory required by the mapping data
     * on faces.
     *
     * If set to a value different from update_general (default), the face
     * information is explicitly built. Currently, MatrixFree supports to
     * cache the following data on faces: inverse Jacobians, Jacobian
     * determinants (JxW), quadrature points, data for Hessians (derivative of
     * Jacobians), and normal vectors.
     */
    UpdateFlags mapping_update_flags_faces_by_cells;

    /**
     * This option can be used to define whether we work on a certain level of
     * the mesh, and not the active cells. If set to invalid_unsigned_int
     * (which is the default value), the active cells are gone through,
     * otherwise the level given by this parameter. Note that if you specify
     * to work on a level, its dofs must be distributed by using
     * <code>dof_handler.distribute_mg_dofs(fe);</code>.
     */
    unsigned int mg_level;

    /**
     * Controls whether to enable reading from vectors without resolving
     * constraints, i.e., just read the local values of the vector. By
     * default, this option is enabled. In case you want to use
     * FEEvaluationBase::read_dof_values_plain, this flag needs to be set.
     */
    bool store_plain_indices;

    /**
     * Option to control whether the indices stored in the DoFHandler
     * should be read and the pattern for task parallelism should be
     * set up in the initialize method of MatrixFree. The default
     * value is true. Can be disabled in case the mapping should be
     * recomputed (e.g. when using a deforming mesh described through
     * MappingEulerian) but the topology of cells has remained the
     * same.
     */
    bool initialize_indices;

    /**
     * Option to control whether the mapping information should be
     * computed in the initialize method of MatrixFree. The default
     * value is true. Can be disabled when only some indices should be
     * set up (e.g. when only a set of independent cells should be
     * computed).
     */
    bool initialize_mapping;

    /**
     * Option to control whether the loops should overlap communications and
     * computations as far as possible in case the vectors passed to the loops
     * support non-blocking data exchange. In most situations, overlapping is
     * faster in case the amount of data to be sent is more than a few
     * kilobytes. If less data is sent, the communication is latency bound on
     * most clusters (point-to-point latency is around 1 microsecond on good
     * clusters by 2016 standards). Depending on the MPI implementation and
     * the fabric, it may be faster to not overlap and wait for the data to
     * arrive. The default is true, i.e., communication and computation are
     * overlapped.
     */
    bool overlap_communication_computation;

    /**
     * By default, the face part will only hold those faces (and ghost
     * elements behind faces) that are going to be processed locally. In case
     * MatrixFree should have access to all neighbors on locally owned cells,
     * this option enables adding the respective faces at the end of the face
     * range.
     */
    bool hold_all_faces_to_owned_cells;

    /**
     * This data structure allows to assign a fraction of cells to different
     * categories when building the information for vectorization. It is used
     * implicitly when working with hp-adaptivity but can also be useful in
     * other contexts, such as in local time stepping where one would like to
     * control which elements together form a batch of cells.
     *
     * This array is accessed by the number given by cell->active_cell_index()
     * when working on the active cells with @p mg_level set to numbers::invalid_unsigned_int and
     * by cell->index() for the level cells.
     *
     * @note This field is empty upon construction of AdditionalData. It is
     * the responsibility of the user to resize this field to
     * `triangulation.n_active_cells()` or `triangulation.n_cells(level)` when
     * filling data.
     */
    std::vector<unsigned int> cell_vectorization_category;

    /**
     * By default, the different categories in @p cell_vectorization_category
     * can be mixed and the algorithm is allowed to merge lower categories with
     * the next higher categories if it is necessary inside the algorithm. This
     * gives a better utilization of the vectorization but might need special
     * treatment, in particular for face integrals. If set to @p true, the
     * algorithm will instead keep different categories separate and not mix
     * them in a single vectorized array.
     */
    bool cell_vectorization_categories_strict;

    /**
     * Assert that vectors passed to the MatrixFree loops are not ghosted.
     * This variable is primarily intended to reveal bugs or performance
     * problems caused by vectors that are involuntarily in ghosted mode,
     * by adding a check that this is not the case. In terms of correctness,
     * the MatrixFree::loop() and MatrixFree::cell_loop() methods support
     * both cases and perform similar operations. In particular, ghost values
     * are always updated on the source vector within the loop, and the
     * difference is only in whether the initial non-ghosted state is restored.
     */
    bool allow_ghosted_vectors_in_loops;

    /**
     * Shared-memory MPI communicator. Default: MPI_COMM_SELF.
     */
    MPI_Comm communicator_sm;
  };

  /**
   * @name 1: Construction and initialization
   */
  //@{
  /**
   * Default empty constructor. Does nothing.
   */
  MatrixFreeFEPoint();

  /**
   * Copy constructor, calls copy_from
   */
  MatrixFreeFEPoint(
    const MatrixFreeFEPoint<dim, Number, VectorizedArrayType> &other);

  /**
   * Destructor.
   */
  ~MatrixFreeFEPoint() override = default;

  /**
   * Extracts the information needed to perform loops over cells. The
   * DoFHandler and AffineConstraints objects describe the layout of degrees
   * of freedom, the DoFHandler and the mapping describe the
   * transformations from unit to real cell, and the finite element
   * underlying the DoFHandler together with the quadrature formula
   * describe the local operations. Note that the finite element underlying
   * the DoFHandler must either be scalar or contain several copies of the
   * same element. Mixing several different elements into one FESystem is
   * not allowed. In that case, use the initialization function with
   * several DoFHandler arguments.
   */
  template <typename QuadratureType, typename number2, typename MappingType>
  void
  reinit(const MappingType &               mapping,
         const DoFHandler<dim> &           dof_handler,
         const AffineConstraints<number2> &constraint,
         const QuadratureType &            quad,
         const AdditionalData &            additional_data = AdditionalData());


  /**
   * Extracts the information needed to perform loops over cells. The
   * DoFHandler and AffineConstraints objects describe the layout of degrees of
   * freedom, the DoFHandler and the mapping describe the transformations from
   * unit to real cell, and the finite element underlying the DoFHandler
   * together with the quadrature formula describe the local operations. As
   * opposed to the scalar case treated with the other initialization
   * functions, this function allows for problems with two or more different
   * finite elements. The DoFHandlers to each element must be passed as
   * pointers to the initialization function. Alternatively, a system of
   * several components may also be represented by a single DoFHandler with an
   * FESystem element. The prerequisite for this case is that each base
   * element of the FESystem must be compatible with the present class, such
   * as the FE_Q or FE_DGQ classes.
   *
   * This function also allows for using several quadrature formulas, e.g.
   * when the description contains independent integrations of elements of
   * different degrees. However, the number of different quadrature formulas
   * can be sets independently from the number of DoFHandlers, when several
   * elements are always integrated with the same quadrature formula.
   */
  template <typename QuadratureType, typename number2, typename MappingType>
  void
  reinit(const MappingType &                                    mapping,
         const std::vector<const DoFHandler<dim> *> &           dof_handler,
         const std::vector<const AffineConstraints<number2> *> &constraint,
         const std::vector<QuadratureType> &                    quad,
         const AdditionalData &additional_data = AdditionalData());


  /**
   * Initializes the data structures. Same as before, but now the index set
   * description of the locally owned range of degrees of freedom is taken
   * from the DoFHandler. Moreover, only a single quadrature formula is used,
   * as might be necessary when several components in a vector-valued problem
   * are integrated together based on the same quadrature formula.
   */
  template <typename QuadratureType, typename number2, typename MappingType>
  void
  reinit(const MappingType &                                    mapping,
         const std::vector<const DoFHandler<dim> *> &           dof_handler,
         const std::vector<const AffineConstraints<number2> *> &constraint,
         const QuadratureType &                                 quad,
         const AdditionalData &additional_data = AdditionalData());


  /**
   * Copy function. Creates a deep copy of all data structures. It is usually
   * enough to keep the data for different operations once, so this function
   * should not be needed very often.
   */
  void
  copy_from(const MatrixFreeFEPoint<dim, Number, VectorizedArrayType>
              &matrix_free_base);

  /**
   * Refreshes the geometry data stored in the MappingInfo fields when the
   * underlying geometry has changed (e.g. by a mapping that can deform
   * through a change in the spatial configuration like MappingFEField)
   * whereas the topology of the mesh and unknowns have remained the
   * same. Compared to reinit(), this operation only has to re-generate the
   * geometry arrays and can thus be significantly cheaper (depending on the
   * cost to evaluate the geometry).
   */
  void
  update_mapping(const Mapping<dim> &mapping);

  /**
   * Same as above but with hp::MappingCollection.
   */
  void
  update_mapping(const std::shared_ptr<hp::MappingCollection<dim>> &mapping);

  /**
   * Clear all data fields and brings the class into a condition similar to
   * after having called the default constructor.
   */
  void
  clear();

  //@}

  /**
   * This class defines the type of data access for face integrals in loop ()
   * that is passed on to the `update_ghost_values` and `compress` functions
   * of the parallel vectors, with the purpose of being able to reduce the
   * amount of data that must be exchanged. The data exchange is a real
   * bottleneck in particular for high-degree DG methods, therefore a more
   * restrictive way of exchange is clearly beneficial. Note that this
   * selection applies to FEFaceEvaluation objects assigned to the exterior
   * side of cells accessing `FaceToCellTopology::exterior_cells` only; all
   * <i>interior</i> objects are available in any case.
   */
  enum class DataAccessOnFaces
  {
    /**
     * The loop does not involve any FEFaceEvaluation access into neighbors,
     * as is the case with only boundary integrals (but no interior face
     * integrals) or when doing mass matrices in a MatrixFree::cell_loop()
     * like setup.
     */
    none,

    /**
     * The loop does only involve FEFaceEvaluation access into neighbors by
     * function values, such as FEFaceEvaluation::gather_evaluate() with
     * argument EvaluationFlags::values, but no access to shape function
     * derivatives (which typically need to access more data). For FiniteElement
     * types where only some of the shape functions have support on a face, such
     * as an FE_DGQ element with Lagrange polynomials with nodes on the element
     * surface, the data exchange is reduced from `(k+1)^dim` to
     * `(k+1)^(dim-1)`.
     */
    values,

    /**
     * Same as above. To be used if data has to be accessed from exterior faces
     * if FEFaceEvaluation was reinitialized by providing the cell batch number
     * and a face number. This configuration is useful in the context of
     * cell-centric loops.
     *
     * @pre AdditionalData::hold_all_faces_to_owned_cells has to enabled.
     */
    values_all_faces,

    /**
     * The loop does involve FEFaceEvaluation access into neighbors by
     * function values and gradients, but no second derivatives, such as
     * FEFaceEvaluation::gather_evaluate() with EvaluationFlags::values and
     * EvaluationFlags::gradients set. For FiniteElement types where only some
     * of the shape functions have non-zero value and first derivative on a
     * face, such as an FE_DGQHermite element, the data exchange is reduced,
     * e.g. from `(k+1)^dim` to `2(k+1)^(dim-1)`. Note that for bases that do
     * not have this special property, the full neighboring data is sent anyway.
     */
    gradients,

    /**
     * Same as above. To be used if data has to be accessed from exterior faces
     * if FEFaceEvaluation was reinitialized by providing the cell batch number
     * and a face number. This configuration is useful in the context of
     * cell-centric loops.
     *
     * @pre AdditionalData::hold_all_faces_to_owned_cells has to enabled.
     */
    gradients_all_faces,

    /**
     * General setup where the user does not want to make a restriction. This
     * is typically more expensive than the other options, but also the most
     * conservative one because the full data of elements behind the faces to
     * be computed locally will be exchanged.
     */
    unspecified
  };


  /**
   * In the hp-adaptive case, a subrange of cells as computed during the cell
   * loop might contain elements of different degrees. Use this function to
   * compute what the subrange for an individual finite element degree is. The
   * finite element degree is associated to the vector component given in the
   * function call.
   */
  std::pair<unsigned int, unsigned int>
  create_cell_subrange_hp(const std::pair<unsigned int, unsigned int> &range,
                          const unsigned int fe_degree,
                          const unsigned int dof_handler_index = 0) const;

  /**
   * In the hp-adaptive case, a subrange of cells as computed during the cell
   * loop might contain elements of different degrees. Use this function to
   * compute what the subrange for a given index the hp-finite element, as
   * opposed to the finite element degree in the other function.
   */
  std::pair<unsigned int, unsigned int>
  create_cell_subrange_hp_by_index(
    const std::pair<unsigned int, unsigned int> &range,
    const unsigned int                           fe_index,
    const unsigned int                           dof_handler_index = 0) const;

  /**
   * In the hp-adaptive case, return number of active FE indices.
   */
  unsigned int
  n_active_fe_indices() const;

  /**
   * In the hp-adaptive case, return the active FE index of a cell range.
   */
  unsigned int
  get_cell_active_fe_index(
    const std::pair<unsigned int, unsigned int> range) const;

  /**
   * In the hp-adaptive case, return the active FE index of a face range.
   */
  unsigned int
  get_face_active_fe_index(const std::pair<unsigned int, unsigned int> range,
                           const bool is_interior_face = true) const;

  //@}

  /**
   * @name 3: Initialization of vectors
   */
  //@{
  /**
   * Initialize function for a vector with each entry associated with a cell
   * batch (cell data). For reading and writing the vector use:
   * FEEvaluationBase::read_cell_data() and FEEvaluationBase::write_cell_data().
   */
  template <typename T>
  void
  initialize_cell_data_vector(AlignedVector<T> &vec) const;

  /**
   * Initialize function for a vector with each entry associated with a face
   * batch (face data). For reading and writing the vector use:
   * FEEvaluationBase::read_face_data() and FEEvaluationBase::write_face_data().
   */
  template <typename T>
  void
  initialize_face_data_vector(AlignedVector<T> &vec) const;

  /**
   * Initialize function for a general serial non-block vector.
   * After a call to this function, the length of the vector is equal to the
   * total number of degrees of freedom in the DoFHandler. Vector entries are
   * initialized with zero.
   *
   * If MatrixFree was set up with several DoFHandler objects, the parameter
   * @p dof_handler_index defines which component is to be used.
   *
   * @note Serial vectors also include Trilinos and PETSc vectors; however
   * in these cases, MatrixFree has to be used in a serial context, i.e., the
   * size of the communicator has to be exactly one.
   */
  template <typename VectorType>
  void
  initialize_dof_vector(VectorType &       vec,
                        const unsigned int dof_handler_index = 0) const;

  /**
   * Specialization of the method initialize_dof_vector() for the
   * class LinearAlgebra::distributed::Vector@<Number@>.
   * See the other function with the same name for the general descriptions.
   *
   * @note For the parallel vectors used with MatrixFree and in FEEvaluation, a
   * vector needs to hold all
   * @ref GlossLocallyActiveDof "locally active DoFs"
   * and also some of the
   * @ref GlossLocallyRelevantDof "locally relevant DoFs".
   * The selection of DoFs is such that one can read all degrees of freedom on
   * all locally relevant elements (locally active) plus the degrees of freedom
   * that constraints expand into from the locally owned cells. However, not
   * all locally relevant DoFs are stored because most of them would never be
   * accessed in matrix-vector products and result in too much data sent
   * around which impacts the performance.
   */
  template <typename Number2>
  void
  initialize_dof_vector(LinearAlgebra::distributed::Vector<Number2> &vec,
                        const unsigned int dof_handler_index = 0) const;

  /**
   * Return the partitioner that represents the locally owned data and the
   * ghost indices where access is needed to for the cell loop. The
   * partitioner is constructed from the locally owned dofs and ghost dofs
   * given by the respective fields. If you want to have specific information
   * about these objects, you can query them with the respective access
   * functions. If you just want to initialize a (parallel) vector, you should
   * usually prefer this data structure as the data exchange information can
   * be reused from one vector to another.
   */
  const std::shared_ptr<const Utilities::MPI::Partitioner> &
  get_vector_partitioner(const unsigned int dof_handler_index = 0) const;

  /**
   * Return the set of cells that are owned by the processor.
   */
  const IndexSet &
  get_locally_owned_set(const unsigned int dof_handler_index = 0) const;

  /**
   * Return the set of ghost cells needed but not owned by the processor.
   */
  const IndexSet &
  get_ghost_set(const unsigned int dof_handler_index = 0) const;

  /**
   * Return a list of all degrees of freedom that are constrained. The list
   * is returned in MPI-local index space for the locally owned range of the
   * vector, not in global MPI index space that spans all MPI processors. To
   * get numbers in global index space, call
   * <tt>get_vector_partitioner()->local_to_global</tt> on an entry of the
   * vector. In addition, it only returns the indices for degrees of freedom
   * that are owned locally, not for ghosts.
   */
  const std::vector<unsigned int> &
  get_constrained_dofs(const unsigned int dof_handler_index = 0) const;

  /**
   * Computes a renumbering of degrees of freedom that better fits with the
   * data layout in MatrixFree according to the given layout of data. Note that
   * this function does not re-arrange the information stored in this class,
   * but rather creates a renumbering for consumption of
   * DoFHandler::renumber_dofs. To have any effect a MatrixFree object must be
   * set up again using the renumbered DoFHandler and AffineConstraints. Note
   * that if a DoFHandler calls DoFHandler::renumber_dofs, all information in
   * MatrixFree becomes invalid.
   */
  void
  renumber_dofs(std::vector<types::global_dof_index> &renumbering,
                const unsigned int                    dof_handler_index = 0);

  //@}

  /**
   * @name 4: General information
   */
  //@{
  /**
   * Return whether a given FiniteElement @p fe is supported by this class.
   */
  template <int spacedim>
  static bool
  is_supported(const FiniteElement<dim, spacedim> &fe);

  /**
   * Return the number of different DoFHandlers specified at initialization.
   */
  unsigned int
  n_components() const;

  /**
   * For the finite element underlying the DoFHandler specified by @p
   * dof_handler_index, return the number of base elements.
   */
  unsigned int
  n_base_elements(const unsigned int dof_handler_index) const;

  /**
   * Return the number of cells this structure is based on. If you are using a
   * usual DoFHandler, it corresponds to the number of (locally owned) active
   * cells. Note that most data structures in this class do not directly act
   * on this number but rather on n_cell_batches() which gives the number of
   * cells as seen when lumping several cells together with vectorization.
   */
  unsigned int
  n_physical_cells() const;

  /**
   * @deprecated Use n_cell_batches() instead.
   */
  DEAL_II_DEPRECATED unsigned int
  n_macro_cells() const;

  /**
   * Return the number of cell batches that this structure works on. The
   * batches are formed by application of vectorization over several cells in
   * general. The cell range in @p cell_loop runs from zero to
   * n_cell_batches() (exclusive), so this is the appropriate size if you want
   * to store arrays of data for all cells to be worked on. This number is
   * approximately `n_physical_cells()/VectorizedArray::%size()`
   * (depending on how many cell batches that do not get filled up completely).
   */
  unsigned int
  n_cell_batches() const;

  /**
   * Return the number of additional cell batches that this structure keeps
   * for face integration. Note that not all cells that are ghosted in the
   * triangulation are kept in this data structure, but only the ones which
   * are necessary for evaluating face integrals from both sides.
   */
  unsigned int
  n_ghost_cell_batches() const;

  /**
   * Return the number of interior face batches that this structure works on.
   * The batches are formed by application of vectorization over several faces
   * in general. The face range in @p loop runs from zero to
   * n_inner_face_batches() (exclusive), so this is the appropriate size if
   * you want to store arrays of data for all interior faces to be worked on.
   * Note that it returns 0 unless mapping_update_flags_inner_faces is set
   * to a value different from  UpdateFlags::update_default.
   */
  unsigned int
  n_inner_face_batches() const;

  /**
   * Return the number of boundary face batches that this structure works on.
   * The batches are formed by application of vectorization over several faces
   * in general. The face range in @p loop runs from n_inner_face_batches() to
   * n_inner_face_batches()+n_boundary_face_batches() (exclusive), so if you
   * need to store arrays that hold data for all boundary faces but not the
   * interior ones, this number gives the appropriate size.
   * Note that it returns 0 unless mapping_update_flags_boundary_faces is set
   * to a value different from UpdateFlags::update_default.
   */
  unsigned int
  n_boundary_face_batches() const;

  /**
   * Return the number of faces that are not processed locally but belong to
   * locally owned faces.
   */
  unsigned int
  n_ghost_inner_face_batches() const;

  /**
   * In order to apply different operators to different parts of the boundary,
   * this method can be used to query the boundary id of a given face in the
   * faces' own sorting by lanes in a VectorizedArray. Only valid for an index
   * indicating a boundary face.
   */
  types::boundary_id
  get_boundary_id(const unsigned int face_batch_index) const;

  /**
   * Return the boundary ids for the faces within a cell, using the cells'
   * sorting by lanes in the VectorizedArray.
   */
  std::array<types::boundary_id, VectorizedArrayType::size()>
  get_faces_by_cells_boundary_id(const unsigned int cell_batch_index,
                                 const unsigned int face_number) const;

  /**
   * Return the DoFHandler with the index as given to the respective
   * `std::vector` argument in the reinit() function.
   */
  const DoFHandler<dim> &
  get_dof_handler(const unsigned int dof_handler_index = 0) const;

  /**
   * Return the DoFHandler with the index as given to the respective
   * `std::vector` argument in the reinit() function. Note that if you want to
   * call this function with a template parameter different than the default
   * one, you will need to use the `template` before the function call, i.e.,
   * you will have something like `matrix_free.template
   * get_dof_handler<hp::DoFHandler<dim>>()`.
   *
   * @deprecated Use the non-templated equivalent of this function.
   */
  template <typename DoFHandlerType>
  DEAL_II_DEPRECATED const DoFHandlerType &
  get_dof_handler(const unsigned int dof_handler_index = 0) const;

  /**
   * Return the cell iterator in deal.II speak to a given cell batch
   * (populating several lanes in a VectorizedArray) and the lane index within
   * the vectorization across cells in the renumbering of this structure.
   *
   * Note that the cell iterators in deal.II go through cells differently to
   * what the cell loop of this class does. This is because several cells are
   * processed together (vectorization across cells), and since cells with
   * neighbors on different MPI processors need to be accessed at a certain
   * time when accessing remote data and overlapping communication with
   * computation.
   */
  typename DoFHandler<dim>::cell_iterator
  get_cell_iterator(const unsigned int cell_batch_index,
                    const unsigned int lane_index,
                    const unsigned int dof_handler_index = 0) const;

  /**
   * This returns the level and index for the cell that would be returned by
   * get_cell_iterator() for the same arguments `cell_batch_index` and
   * `lane_index`.
   */
  std::pair<int, int>
  get_cell_level_and_index(const unsigned int cell_batch_index,
                           const unsigned int lane_index) const;

  /**
   * Return the cell iterator in deal.II speak to an interior/exterior cell of
   * a face in a pair of a face batch and lane index. The second element of
   * the pair is the face number so that the face iterator can be accessed:
   * `pair.first->face(pair.second);`
   *
   * Note that the face iterators in deal.II go through cells differently to
   * what the face/boundary loop of this class does. This is because several
   * faces are worked on together (vectorization), and since faces with neighbor
   * cells on different MPI processors need to be accessed at a certain time
   * when accessing remote data and overlapping communication with computation.
   */
  std::pair<typename DoFHandler<dim>::cell_iterator, unsigned int>
  get_face_iterator(const unsigned int face_batch_index,
                    const unsigned int lane_index,
                    const bool         interior     = true,
                    const unsigned int fe_component = 0) const;

  /**
   * @copydoc MatrixFree::get_cell_iterator()
   *
   * @deprecated Use get_cell_iterator() instead.
   */
  DEAL_II_DEPRECATED typename DoFHandler<dim>::active_cell_iterator
  get_hp_cell_iterator(const unsigned int cell_batch_index,
                       const unsigned int lane_index,
                       const unsigned int dof_handler_index = 0) const;

  /**
   * Since this class uses vectorized data types with usually more than one
   * value in the data field, a situation might occur when some components of
   * the vector type do not correspond to an actual cell in the mesh. When
   * using only this class, one usually does not need to bother about that
   * fact since the values are padded with zeros. However, when this class is
   * mixed with deal.II access to cells, care needs to be taken. This function
   * returns @p true if not all `n_lanes` cells for the given
   * `cell_batch_index` correspond to actual cells of the mesh and some are
   * merely present for padding reasons. To find out how many cells are
   * actually used, use the function n_active_entries_per_cell_batch().
   */
  bool
  at_irregular_cell(const unsigned int cell_batch_index) const;

  /**
   * @deprecated Use n_active_entries_per_cell_batch() instead.
   */
  DEAL_II_DEPRECATED unsigned int
  n_components_filled(const unsigned int cell_batch_number) const;

  /**
   * This query returns how many cells among the `VectorizedArrayType::size()`
   * many cells within a cell batch to actual cells in the mesh, rather than
   * being present for padding reasons. For most given cell batches in
   * n_cell_batches(), this number is equal to `VectorizedArrayType::size()`,
   * but there might be one or a few cell batches in the mesh (where the
   * numbers do not add up) where only some of the cells within a batch are
   * used, indicated by the function at_irregular_cell().
   */
  unsigned int
  n_active_entries_per_cell_batch(const unsigned int cell_batch_index) const;

  /**
   * Use this function to find out how many faces over the length of
   * vectorization data types correspond to real faces (both interior and
   * boundary faces, as those use the same indexing but with different ranges)
   * in the mesh. For most given indices in n_inner_faces_batches() and
   * n_boundary_face_batches(), this is just @p vectorization_length many, but
   * there might be one or a few meshes (where the numbers do not add up)
   * where there are less such lanes filled.
   */
  unsigned int
  n_active_entries_per_face_batch(const unsigned int face_batch_index) const;

  /**
   * Return the number of degrees of freedom per cell for a given hp-index.
   */
  unsigned int
  get_dofs_per_cell(const unsigned int dof_handler_index  = 0,
                    const unsigned int hp_active_fe_index = 0) const;

  /**
   * Return the number of quadrature points per cell for a given hp-index.
   */
  unsigned int
  get_n_q_points(const unsigned int quad_index         = 0,
                 const unsigned int hp_active_fe_index = 0) const;

  /**
   * Return the number of degrees of freedom on each face of the cell for
   * given hp-index.
   */
  unsigned int
  get_dofs_per_face(const unsigned int dof_handler_index  = 0,
                    const unsigned int hp_active_fe_index = 0) const;

  /**
   * Return the number of quadrature points on each face of the cell for
   * given hp-index.
   */
  unsigned int
  get_n_q_points_face(const unsigned int quad_index         = 0,
                      const unsigned int hp_active_fe_index = 0) const;

  /**
   * Return the quadrature rule for given hp-index.
   */
  const Quadrature<dim> &
  get_quadrature(const unsigned int quad_index         = 0,
                 const unsigned int hp_active_fe_index = 0) const;

  /**
   * Return the quadrature rule for given hp-index.
   */
  const Quadrature<dim - 1> &
  get_face_quadrature(const unsigned int quad_index         = 0,
                      const unsigned int hp_active_fe_index = 0) const;

  /**
   * Return the category the current batch range of cells was assigned to.
   * Categories run between the given values in the field
   * AdditionalData::cell_vectorization_category for non-hp-DoFHandler types
   * and return the active FE index in the hp-adaptive case.
   */
  unsigned int
  get_cell_range_category(
    const std::pair<unsigned int, unsigned int> cell_batch_range) const;

  /**
   * Return the category of the cells on the two sides of the current batch
   * range of faces.
   */
  std::pair<unsigned int, unsigned int>
  get_face_range_category(
    const std::pair<unsigned int, unsigned int> face_batch_range) const;

  /**
   * Return the category the current batch of cells was assigned to. Categories
   * run between the given values in the field
   * AdditionalData::cell_vectorization_category for non-hp-DoFHandler types
   * and return the active FE index in the hp-adaptive case.
   */
  unsigned int
  get_cell_category(const unsigned int cell_batch_index) const;

  /**
   * Return the category of the cells on the two sides of the current batch of
   * faces.
   */
  std::pair<unsigned int, unsigned int>
  get_face_category(const unsigned int face_batch_index) const;

  /**
   * Queries whether or not the indexation has been set.
   */
  bool
  indices_initialized() const;

  /**
   * Queries whether or not the geometry-related information for the cells has
   * been set.
   */
  bool
  mapping_initialized() const;

  /**
   * Return the level of the mesh to be worked on. Returns
   * numbers::invalid_unsigned_int if working on active cells.
   */
  unsigned int
  get_mg_level() const;

  /**
   * Return an approximation of the memory consumption of this class in
   * bytes.
   */
  std::size_t
  memory_consumption() const;

  /**
   * Prints a detailed summary of memory consumption in the different
   * structures of this class to the given output stream.
   */
  template <typename StreamType>
  void
  print_memory_consumption(StreamType &out) const;

  /**
   * Prints a summary of this class to the given output stream. It is focused
   * on the indices, and does not print all the data stored.
   */
  void
  print(std::ostream &out) const;

  //@}

  /**
   * @name 5: Access of internal data structure
   *
   * Note: Expert mode, interface not stable between releases.
   */
  //@{
  /**
   * Return information on task graph.
   */
  const internal::MatrixFreeFunctions::TaskInfo &
  get_task_info() const;

  /*
   * Return geometry-dependent information on the cells.
   */
  const internal::MatrixFreeFunctions::
    MappingInfo<dim, Number, VectorizedArrayType> &
    get_mapping_info() const;

  /**
   * Return information on indexation degrees of freedom.
   */
  const internal::MatrixFreeFunctions::DoFInfo &
  get_dof_info(const unsigned int dof_handler_index_component = 0) const;

  /**
   * Return the number of weights in the constraint pool.
   */
  unsigned int
  n_constraint_pool_entries() const;

  /**
   * Return a pointer to the first number in the constraint pool data with
   * index @p pool_index (to be used together with @p constraint_pool_end()).
   */
  const Number *
  constraint_pool_begin(const unsigned int pool_index) const;

  /**
   * Return a pointer to one past the last number in the constraint pool data
   * with index @p pool_index (to be used together with @p
   * constraint_pool_begin()).
   */
  const Number *
  constraint_pool_end(const unsigned int pool_index) const;

  /**
   * Return the unit cell information for given hp-index.
   */
  const internal::MatrixFreeFunctions::ShapeInfo<VectorizedArrayType> &
  get_shape_info(const unsigned int dof_handler_index_component = 0,
                 const unsigned int quad_index                  = 0,
                 const unsigned int fe_base_element             = 0,
                 const unsigned int hp_active_fe_index          = 0,
                 const unsigned int hp_active_quad_index        = 0) const;

  /**
   * Return the connectivity information of a face.
   */
  const internal::MatrixFreeFunctions::FaceToCellTopology<
    VectorizedArrayType::size()> &
  get_face_info(const unsigned int face_batch_index) const;


  /**
   * Return the table that translates a triple of the macro cell number,
   * the index of a face within a cell and the index within the cell batch of
   * vectorization into the index within the faces array.
   */
  const Table<3, unsigned int> &
  get_cell_and_face_to_plain_faces() const;

  /**
   * Obtains a scratch data object for internal use. Make sure to release it
   * afterwards by passing the pointer you obtain from this object to the
   * release_scratch_data() function. This interface is used by FEEvaluation
   * objects for storing their data structures.
   *
   * The organization of the internal data structure is a thread-local storage
   * of a list of vectors. Multiple threads will each get a separate storage
   * field and separate vectors, ensuring thread safety. The mechanism to
   * acquire and release objects is similar to the mechanisms used for the
   * local contributions of WorkStream, see
   * @ref workstream_paper "the WorkStream paper".
   */
  AlignedVector<VectorizedArrayType> *
  acquire_scratch_data() const;

  /**
   * Makes the object of the scratchpad available again.
   */
  void
  release_scratch_data(const AlignedVector<VectorizedArrayType> *memory) const;

  /**
   * Obtains a scratch data object for internal use. Make sure to release it
   * afterwards by passing the pointer you obtain from this object to the
   * release_scratch_data_non_threadsafe() function. Note that, as opposed to
   * acquire_scratch_data(), this method can only be called by a single thread
   * at a time, but opposed to the acquire_scratch_data() it is also possible
   * that the thread releasing the scratch data can be different than the one
   * that acquired it.
   */
  AlignedVector<Number> *
  acquire_scratch_data_non_threadsafe() const;

  /**
   * Makes the object of the scratch data available again.
   */
  void
  release_scratch_data_non_threadsafe(
    const AlignedVector<Number> *memory) const;

  //@}

private:
  /**
   * This is the actual reinit function that sets up the indices for the
   * DoFHandler case.
   */
  template <typename number2, int q_dim>
  void
  internal_reinit(
    const std::shared_ptr<hp::MappingCollection<dim>> &    mapping,
    const std::vector<const DoFHandler<dim, dim> *> &      dof_handlers,
    const std::vector<const AffineConstraints<number2> *> &constraint,
    const std::vector<IndexSet> &                          locally_owned_set,
    const std::vector<hp::QCollection<q_dim>> &            quad,
    const AdditionalData &                                 additional_data);

  /**
   * Initializes the fields in DoFInfo together with the constraint pool that
   * holds all different weights in the constraints (not part of DoFInfo
   * because several DoFInfo classes can have the same weights which
   * consequently only need to be stored once).
   */
  template <typename number2>
  void
  initialize_indices(
    const std::vector<const AffineConstraints<number2> *> &constraint,
    const std::vector<IndexSet> &                          locally_owned_set,
    const AdditionalData &                                 additional_data);

  /**
   * Initializes the DoFHandlers based on a DoFHandler<dim> argument.
   */
  void
  initialize_dof_handlers(
    const std::vector<const DoFHandler<dim, dim> *> &dof_handlers,
    const AdditionalData &                           additional_data);

  /**
   * Pointers to the DoFHandlers underlying the current problem.
   */
  std::vector<SmartPointer<const DoFHandler<dim>>> dof_handlers;

  /**
   * Contains the information about degrees of freedom on the individual cells
   * and constraints.
   */
  std::vector<internal::MatrixFreeFunctions::DoFInfo> dof_info;

  /**
   * Contains the weights for constraints stored in DoFInfo. Filled into a
   * separate field since several vector components might share similar
   * weights, which reduces memory consumption. Moreover, it obviates template
   * arguments on DoFInfo and keeps it a plain field of indices only.
   */
  std::vector<Number> constraint_pool_data;

  /**
   * Contains an indicator to the start of the ith index in the constraint
   * pool data.
   */
  std::vector<unsigned int> constraint_pool_row_index;

  /**
   * Holds information on transformation of cells from reference cell to real
   * cell that is needed for evaluating integrals.
   */
  internal::MatrixFreeFunctions::MappingInfo<dim, Number, VectorizedArrayType>
    mapping_info;

  /**
   * Contains shape value information on the unit cell.
   */
  Table<4, internal::MatrixFreeFunctions::ShapeInfo<VectorizedArrayType>>
    shape_info;

  /**
   * Describes how the cells are gone through. With the cell level (first
   * index in this field) and the index within the level, one can reconstruct
   * a deal.II cell iterator and use all the traditional things deal.II offers
   * to do with cell iterators.
   */
  std::vector<std::pair<unsigned int, unsigned int>> cell_level_index;


  /**
   * For discontinuous Galerkin, the cell_level_index includes cells that are
   * not on the local processor but that are needed to evaluate the cell
   * integrals. In cell_level_index_end_local, we store the number of local
   * cells.
   */
  unsigned int cell_level_index_end_local;

  /**
   * Stores the basic layout of the cells and faces to be treated, including
   * the task layout for the shared memory parallelization and possible
   * overlaps between communications and computations with MPI.
   */
  internal::MatrixFreeFunctions::TaskInfo task_info;

  /**
   * Vector holding face information. Only initialized if
   * build_face_info=true.
   */
  internal::MatrixFreeFunctions::FaceInfo<VectorizedArrayType::size()>
    face_info;

  /**
   * Stores whether indices have been initialized.
   */
  bool indices_are_initialized;

  /**
   * Stores whether indices have been initialized.
   */
  bool mapping_is_initialized;

  /**
   * Scratchpad memory for use in evaluation. We allow more than one
   * evaluation object to attach to this field (this, the outer
   * std::vector), so we need to keep tracked of whether a certain data
   * field is already used (first part of pair) and keep a list of
   * objects.
   */
  mutable Threads::ThreadLocalStorage<
    std::list<std::pair<bool, AlignedVector<VectorizedArrayType>>>>
    scratch_pad;

  /**
   * Scratchpad memory for use in evaluation and other contexts, non-thread
   * safe variant.
   */
  mutable std::list<std::pair<bool, AlignedVector<Number>>>
    scratch_pad_non_threadsafe;

  /**
   * Stored the level of the mesh to be worked on.
   */
  unsigned int mg_level;
};



/*----------------------- Inline functions ----------------------------------*/

#ifndef DOXYGEN



#endif // ifndef DOXYGEN



DEAL_II_NAMESPACE_CLOSE

#endif
