#ifndef dealii_non_matching_cut_mesh_classifier
#define dealii_non_matching_cut_mesh_classifier

#include <deal.II/base/config.h>

#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_handler.h>

#include <deal.II/grid/tria.h>
#include <deal.II/grid/tria_accessor.h>

#include <cstdio>
#include <memory>

DEAL_II_NAMESPACE_OPEN
namespace NonMatching
{
  /**
   * Type describing how a cell or a face is located relative to the zero
   * contour of a level set function, $\psi$. The values of the type correspond
   * to:
   *
   * INSIDE        if $\psi(x) < 0$,
   * OUTSIDE       if $\psi(x) > 0$,
   * INTERSECTED   if $\psi(x)$ varies in sign,
   * over the cell/face.
   *
   * UNASSIGNED is used to describe that a cell/face hasn't been classified
   * as one of the other types.
   */
  enum class LocationToLevelSet
  {
    INSIDE,
    OUTSIDE,
    INTERSECTED,
    UNASSIGNED
  };



  namespace internal
  {
    namespace CutMeshClassifierImplementation
    {
      /**
       * Struct which stores the locations relative to the level set function
       * for a single cell and its faces.
       */
      template <int dim>
      struct CellAndFaceLocations
      {
        /**
         * Constructor, sets all locations to UNASSIGNED.
         */
        CellAndFaceLocations();

        /**
         * Location of the cell.
         */
        LocationToLevelSet cell_location;

        /**
         * Array storing the location of each face of the cell.
         */
        std::array<LocationToLevelSet, GeometryInfo<dim>::faces_per_cell>
          face_locations;
      };


      /**
       * An interface which defines an algorithm for how to classify individual
       * cells and faces into the categories in LocationToLevelSet.
       *
       * This interface is needed because we want to vary the algorithm which is
       * used depending on how the level set function is described. In
       * particular we want to use different concrete implementations when the
       * level set function is described by a pair {DofHandler<dim>, Vector} and
       * when it is described by a Function<dim>.
       */
      template <int dim>
      class CellFaceClassifier
      {
      public:
        /**
         * Destructor, declared just to mark it virtual.
         */
        virtual ~CellFaceClassifier() = default;

        /**
         * Classifies a given face of the incoming cell into one of the
         * categories of LocationToLevelSet and returns the result.
         */
        virtual LocationToLevelSet
        determine_face_location_to_levelset(
          const typename Triangulation<dim>::active_cell_iterator &cell,
          const unsigned int face_index) = 0;

        /**
         * Classifies the incoming cell into one of the categories of
         * LocationToLevelSet and returns the result.
         */
        virtual LocationToLevelSet
        determine_cell_location_to_levelset(
          const typename Triangulation<dim>::active_cell_iterator &cell,
          const std::array<LocationToLevelSet,
                           GeometryInfo<dim>::faces_per_cell>
            &face_locations) = 0;
      };

    } // namespace CutMeshClassifierImplementation
  }   // namespace internal


  /**
   * Class responsible for determining how the cells and faces of a
   * triangulation relates to a level set function. Using the reclassify()
   * function the cells/faces of the triangulation are classified into one of
   * the categories of the enum LocationToLevelSet. This described each cell and
   * face as one of the categories inside, outside or intersected depending on
   * how they are located relative to the level set function. This information
   * is required in immersed finite element methods, both when distributing
   * degrees of freedom over the triangulation and when the system is assembled.
   */
  template <int dim>
  class CutMeshClassifier : public Subscriptor
  {
  public:
    /**
     * Constructor. Takes the triangulation that should be classified and
     * and a discrete level set function. The level set function is described as
     * a vector and an associated DoFHandler. The triangulation associated with
     * the DoFHandler must be the same as the first argument.
     */
    template <class VECTOR>
    CutMeshClassifier(const Triangulation<dim> &triangulation,
                      const DoFHandler<dim> &   level_set_dof_handler,
                      const VECTOR &            level_set);


    /**
     * Constructor. Takes the triangulation that should be classified and a
     * level set function described as a function.
     */
    CutMeshClassifier(const Triangulation<dim> &triangulation,
                      const Function<dim> &     level_set);

    /**
     * Perform the classification of the triangulation passed to the
     * constructor.
     */
    void
    reclassify();

    /**
     * Returns how the incoming cell is located relative to the level set
     * function.
     */
    LocationToLevelSet
    location_to_level_set(
      const typename Triangulation<dim>::cell_iterator &cell) const;

    /**
     * Returns how a face of the incoming cell is located relative to the level
     * set function.
     */
    LocationToLevelSet
    location_to_level_set(
      const typename Triangulation<dim>::cell_iterator &cell,
      const unsigned int                                face_index) const;

  private:
    /**
     * Pointer to the triangulation that should be classified.
     */
    const SmartPointer<const Triangulation<dim>> triangulation;

    /**
     * Pointer to the class that is responsible for classifying individual
     * cells/faces. This will be different depending on if the level set
     * function is discrete (descibed as a DoFHandler<dim> and a vector) or
     * described by a Function<dim>.
     */
    const std::unique_ptr<
      internal::CutMeshClassifierImplementation::CellFaceClassifier<dim>>
      cell_face_classifier;

    /**
     * A map that for each cell stores how the cell and its faces are
     * located relative to the level set function.
     */
    std::map<
      typename Triangulation<dim>::cell_iterator,
      internal::CutMeshClassifierImplementation::CellAndFaceLocations<dim>>
      categories;
  };


} // namespace NonMatching
DEAL_II_NAMESPACE_CLOSE

#endif
